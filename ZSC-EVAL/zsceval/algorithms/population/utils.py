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
    2. Maintains partner obs/action history and runs PartnerPredictionNet each step
       to compute pred_context.

    Partner history is updated by the runner via record_partner() before each step.
    Only PPO image obs (ndim==3, shape H×W×C) are recorded; BC featurized obs are
    skipped and pred_context falls back to None for that agent.
    """

    def __init__(self, args, actor, pred_net):
        # EvalPolicy expects a policy object; provide a shim that delegates to actor.
        super().__init__(args, _ActorShim(actor))
        self._actor = actor
        self._pred_net = pred_net
        self._history_len = pred_net.history_len
        self._obs_hist: dict = {}   # (e, partner_a) -> deque[np.ndarray (H,W,C)]
        self._act_hist: dict = {}   # (e, partner_a) -> deque[int]

    # ------------------------------------------------------------------
    def reset(self, num_envs, num_agents):
        super().reset(num_envs, num_agents)
        self._obs_hist.clear()
        self._act_hist.clear()

    # ------------------------------------------------------------------
    def record_partner(self, all_obs, prev_actions, my_agents):
        """Called by the runner at the start of each env step.

        all_obs      : dict (e, a) -> np.ndarray   — obs for every agent
        prev_actions : dict (e, a) -> int           — actions from previous step
        my_agents    : set of (e, a) controlled by this policy
        """
        for ea, obs in all_obs.items():
            if ea in my_agents:
                continue
            # Only use PPO image obs (H×W×C, ndim==3). BC featurized obs are 1-D.
            if obs.ndim != 3:
                continue
            if ea not in self._obs_hist:
                self._obs_hist[ea] = deque(maxlen=self._history_len)
                self._act_hist[ea] = deque(maxlen=self._history_len)
            self._obs_hist[ea].append(obs.astype(np.float32))
            if ea in prev_actions:
                self._act_hist[ea].append(int(prev_actions[ea]))

    # ------------------------------------------------------------------
    def _compute_pred_context(self, agents):
        """Return (N, action_dim) pred_context or None if history is not yet full."""
        contexts = []
        for (e, a) in agents:
            partner_a = 1 - a
            ea_p = (e, partner_a)
            hist_obs = list(self._obs_hist.get(ea_p, []))
            hist_act = list(self._act_hist.get(ea_p, []))
            T = self._history_len
            if len(hist_obs) < T or len(hist_act) < T:
                return None  # any agent lacking history → skip for entire batch
            obs_arr = np.stack(hist_obs[-T:])          # (T, H, W, C)
            act_arr = np.array(hist_act[-T:], dtype=np.int64)  # (T,)
            obs_t = torch.from_numpy(obs_arr).unsqueeze(0)     # (1, T, H, W, C)
            act_t = torch.from_numpy(act_arr).unsqueeze(0)     # (1, T)
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
