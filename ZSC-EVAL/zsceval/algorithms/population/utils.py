from collections import deque

import numpy as np
import torch


def _t2n(x):
    if not isinstance(x, torch.Tensor):
        return x
    return x.detach().cpu().numpy()


class EvalPolicy:
    """A policy for evaluation.
    It maintains hidden states on its own.
    For usage, 'reset' before every eval episode, 'register_control_agents' to indicate agents controlled by this policy and 'step' means an env step.
    """

    def __init__(self, args, policy):
        self.args = args
        self.policy = policy
        self._control_agents = []
        self._map_a2id = dict()

        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self.recurrent_N = args.recurrent_N
        self.hidden_size = args.hidden_size

    @property
    def default_hidden_state(self):
        return np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)

    @property
    def control_agents(self):
        return self._control_agents

    def reset(self, num_envs, num_agents):
        self.num_envs = num_envs
        self.num_agents = num_agents
        self._control_agents = []
        self._map_a2id = dict()
        self._rnn_states = dict()

    def reset_state(self, e, a):
        assert (e, a) in self._control_agents
        self._rnn_states[(e, a)] = self.default_hidden_state

    def register_control_agent(self, e, a):
        if (e, a) not in self._control_agents:
            self._control_agents.append((e, a))
            self._map_a2id[(e, a)] = len(self._control_agents)
            self._rnn_states[(e, a)] = self.default_hidden_state

    def step(self, obs, agents, deterministic=False, masks=None, **kwargs):
        num = len(agents)
        assert obs.shape[0] == num
        rnn_states = [self._rnn_states[ea] for ea in agents]
        if masks is None:
            masks = np.ones((num, 1), dtype=np.float32)
        action, rnn_states = self.policy.act(
            obs, np.stack(rnn_states, axis=0), masks, deterministic=deterministic, **kwargs
        )
        for ea, rnn_state in zip(agents, _t2n(rnn_states)):
            self._rnn_states[ea] = rnn_state
        return _t2n(action)

    def to(self, device):
        self.policy.to(device)

    def prep_rollout(self):
        self.policy.prep_rollout()


class _ActorShim:
    """Minimal shim so EvalPolicy.__init__ receives an object with to()/prep_rollout().
    PH2EvalPolicy holds the real actor directly and never calls through this shim."""

    def __init__(self, actor):
        self._actor = actor

    def to(self, device):
        self._actor.to(device)

    def prep_rollout(self):
        self._actor.eval()


class PH2EvalPolicy(EvalPolicy):
    """EvalPolicy for the PH2 ind policy.

    Differs from EvalPolicy in two ways:
    1. Calls PH2Actor.forward() directly (bypassing PH2Policy.act() which does not
       forward pred_context to the actor) so that pred_context can be injected.
    2. Maintains ego's own obs/action history and runs PartnerPredictionNet each step
       to compute pred_context.

    History is stored as fixed-size numpy arrays initialized to zeros — identical to
    training. pred_context is always computed (never None), matching train behavior
    where episode-start history is zero-padded.

    Runner calls update_obs_hist() BEFORE step() and update_act_hist() AFTER env.step().
    """

    def __init__(self, args, actor, pred_net):
        # EvalPolicy expects a policy object; provide a shim that delegates to actor.
        super().__init__(args, _ActorShim(actor))
        self._actor = actor
        self._pred_net = pred_net
        self._history_len = pred_net.history_len
        # (e, a) -> np.ndarray; shape lazily set on first obs
        self._obs_hist: dict = {}   # (e, a) -> (T, H, W, C)  float32
        self._act_hist: dict = {}   # (e, a) -> (T,)           int64
        self._obs_shape = None      # inferred on first obs

    # ------------------------------------------------------------------
    def reset(self, num_envs, num_agents):
        super().reset(num_envs, num_agents)
        self._obs_hist.clear()
        self._act_hist.clear()

    def _ensure_hist(self, ea, obs_shape):
        """Lazily allocate zero-initialised history arrays for agent ea."""
        if ea not in self._obs_hist:
            T = self._history_len
            self._obs_hist[ea] = np.zeros((T, *obs_shape), dtype=np.float32)
            self._act_hist[ea] = np.zeros((T,),            dtype=np.int64)

    # ------------------------------------------------------------------
    def update_obs_hist(self, all_obs, my_agents):
        """Called by the runner BEFORE policy.step() — roll and append obs_t.

        Matches training: obs_hist includes obs_t when pred_ctx is computed.
        """
        for ea in my_agents:
            obs = all_obs.get(ea)
            if obs is None or obs.ndim != 3:
                continue
            self._ensure_hist(ea, obs.shape)
            self._obs_hist[ea] = np.roll(self._obs_hist[ea], -1, axis=0)
            self._obs_hist[ea][-1] = obs.astype(np.float32)

    def update_act_hist(self, act_dict, my_agents):
        """Called by the runner AFTER env.step() — roll and append act_t.

        Matches training: act_hist excludes act_t when pred_ctx is computed.
        """
        for ea in my_agents:
            if ea not in act_dict:
                continue
            if ea not in self._act_hist:
                # allocate without obs_shape (obs_hist may not exist yet)
                self._act_hist[ea] = np.zeros((self._history_len,), dtype=np.int64)
            self._act_hist[ea] = np.roll(self._act_hist[ea], -1, axis=0)
            self._act_hist[ea][-1] = int(act_dict[ea])

    # ------------------------------------------------------------------
    def _compute_pred_context(self, agents):
        """Return (N, action_dim) pred_context.

        Always returns a value (zero-padded history for episode start),
        matching training where obs_hist/act_hist are zero-initialised.
        Returns None only if obs_hist has not been allocated yet (first step
        before any update_obs_hist call, which should not happen in practice).
        """
        contexts = []
        for ea in agents:
            if ea not in self._obs_hist:
                return None  # guard: update_obs_hist not yet called
            obs_arr = self._obs_hist[ea]                    # (T, H, W, C)
            act_arr = self._act_hist[ea]                    # (T,)
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)  # (1, T, H, W, C)
            act_t = torch.from_numpy(act_arr).unsqueeze(0)  # (1, T)
            dev = next(self._pred_net.parameters()).device
            with torch.no_grad():
                ctx = self._pred_net(obs_t.to(dev), act_t.to(dev))  # (1, act_dim)
            contexts.append(ctx.cpu().numpy())
        if not contexts:
            return None
        return np.concatenate(contexts, axis=0)  # (N, act_dim)

    # ------------------------------------------------------------------
    def step(self, obs, agents, deterministic=False, masks=None, **kwargs):
        """Call PH2Actor.forward() directly so pred_context is forwarded."""
        pred_context = self._compute_pred_context(agents)

        num = len(agents)
        rnn_states = np.stack([self._rnn_states[ea] for ea in agents], axis=0)
        if masks is None:
            masks = np.ones((num, 1), dtype=np.float32)

        obs_t = torch.FloatTensor(obs)
        rnn_t = torch.FloatTensor(rnn_states)
        mask_t = torch.FloatTensor(masks)
        pred_t = torch.FloatTensor(pred_context) if pred_context is not None else None

        avail = kwargs.get("available_actions")
        avail_t = torch.FloatTensor(avail) if avail is not None else None

        with torch.no_grad():
            actions, _, rnn_new = self._actor(
                obs_t, rnn_t, mask_t,
                available_actions=avail_t,
                deterministic=deterministic,
                pred_context=pred_t,
            )
        actions = _t2n(actions)
        rnn_new = _t2n(rnn_new)
        for ea, rnn_state in zip(agents, rnn_new):
            self._rnn_states[ea] = rnn_state
        return actions

    # ------------------------------------------------------------------
    def to(self, device):
        self._actor.to(device)
        self._pred_net.to(device)

    def prep_rollout(self):
        self._actor.eval()
        self._pred_net.eval()
