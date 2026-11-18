import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim

from models.policy import StepEmbedPolicy

# ---------------------------------------------------------------
# Hyperparameters (safe defaults; override via CLI in run_tc_sweep.py)
# ---------------------------------------------------------------
BATCH        = 16         # Default number of trajectories per training batch
LR_NN        = 1e-3       # Default learning rate
LR_LOGZ_MULT = 50.0       # multiplty the learning rate for Trajectory Balance by LR_LOGZ_MULT (lr TB is LR_NN*LR_LOGZ_MULT)
EPS_START    = 0.8        # Initial exploration mixture weight
EPS_END      = 0.05       # Final exploration mixture weight
BETA_START   = 0.3        # Initial reward multiplier in logR = beta * (-E/T)
BETA_END     = 1.0        # Final reward multiplier: full thermal target
ENTROPY_COEF = 0.05       # Initial entropy-bonus weight, gradually reduced to zero during training
LOG_PB_FIXED = 0.0        # For backward probabilities, this graph has unique backward parents. So backward probabilities are 1. 


# =====================================================
#  Batched GFlowNet base class (shared by TB and KC)
# =====================================================
class BatchedGFlowNetBase:
    """
    Shared machinery for batched, weight-tied-across-timesteps GFlowNet
    training: policy construction, the mixture-exploration trajectory
    sampler, and the WITH-grad replay used to compute log p_F and
    entropy for a fixed action sequence.

    TBGFlowNet and KCGFlowNet differ only in loss():
      - TB fits an explicit log_Z parameter against the reward.
      - KC eliminates log_Z via a telescoping identity between pairs of
        trajectories -- the per-trajectory residual log_pf - log_reward
        should be the SAME constant (that constant is log_Z, but it's
        never fit directly) for every trajectory at convergence, so KC's
        loss is just the variance of that residual across the batch,
        and needs no log_Z parameter at all.

    Fixes relative to the original per-trajectory implementation
    (applies to both TB and KC, since both build on this base class):

    1. BATCHING (the dominant cost at large D): sampling and log_pf/
       entropy both advance one position at a time, but process the
       WHOLE batch of trajectories at that position in a single forward
       call, instead of looping over trajectories with batch-size-1
       calls. Forward passes per training step go from O(B*D) to O(D).
    2. Beta no longer scales the policy logits (previously the same
       parameters meant a different effective policy at every point in
       training, and the reward term wasn't tempered by the same beta,
       a real mismatch in the original TB residual). Beta now only
       tempers the REWARD: logR = beta * env.log_reward(...).
    3. Entropy coefficient anneals linearly to zero instead of staying
       fixed for the whole run, so training actually converges to the
       true Boltzmann target rather than a permanently-entropy-
       regularized approximation of it.
    4. Exploration samples directly from the mixture
       q = (1-eps)*pi + eps/n_actions, rather than an if/else branch --
       same behavior, batchable, and log_pf still evaluates log pi_theta
       of whatever action was drawn (which is what makes off-policy
       exploration valid here in the first place -- no importance
       weighting needed, since both TB's and KC's fixed points don't
       depend on the behavior policy as long as it has full support).
    5. Reward uses env.energy_batch (vectorized) instead of one
       env.energy call per sample in a Python loop.
    """
    def __init__(self, env, T, device="cpu", init_state_dict=None,
                 entropy_coef0=ENTROPY_COEF, entropy_anneal_frac=0.8):
        self.env = env
        self.T = T
        self.device = device
        grid_N = env.N if env.kind in ("ising", "ea") else None
        self.policy = StepEmbedPolicy(env.D, grid_N=grid_N).to(device)
        if init_state_dict is not None:
            self.policy.load_state_dict(init_state_dict)
        self.entropy_coef0 = entropy_coef0
        self.entropy_anneal_frac = entropy_anneal_frac
        self.opt = None  # subclasses build this once all params are registered

    def state_dict(self):
        return self.policy.state_dict()

    def schedule(self, it, n_iter):
        frac = min(it / n_iter, 1.0)
        beta = BETA_START + frac * (BETA_END - BETA_START)
        eps = EPS_START + frac * (EPS_END - EPS_START)
        anneal_t = max(self.entropy_anneal_frac * n_iter, 1)
        ent_coef = self.entropy_coef0 * max(0.0, 1.0 - it / anneal_t)
        return beta, eps, ent_coef


    @torch.no_grad()
    def _sample_actions_batch(self, B, eps):
        """No-grad rollout: advance B trajectories together, one position
        at a time, sampling from the mixture q = (1-eps)*pi + eps/A.
        Returns the recorded actions [B, D] (the terminal spins can be
        cheaply replayed from these via _replay_batch, which is also
        where gradients get attached for the loss)."""
        D = self.env.D
        spins = torch.zeros(B, D, device=self.device)
        mask = torch.zeros(B, D, device=self.device)
        actions = torch.zeros(B, D, dtype=torch.long, device=self.device)
        for t in range(D):
            logits = self.policy(spins, mask, t)
            #pi = torch.softmax(logits, dim=-1)

            tilt_strength = getattr(self, "tilt_strength", 1.0) #tilt_strength = 1.0  # Hyperparameter: 0 disables tilt.
            candidates = spins[:, None, :].expand(-1, 2, -1).clone()
            candidates[:, :, t] = spins.new_tensor([-1.0, 1.0])
            energy_after = self.env.energy_batch(candidates.reshape(-1, self.env.D)).reshape(-1, 2)
            energy_before = self.env.energy_batch(spins).unsqueeze(1)
            transition_scores = -(energy_after - energy_before) / self.T

            pi = torch.softmax(logits + tilt_strength * transition_scores, dim=-1)
            q = (1.0 - eps) * pi + eps / pi.shape[-1]
            a = torch.multinomial(q, 1).squeeze(-1)
            actions[:, t] = a
            spins[:, t] = torch.where(a == 0, -1.0, 1.0)
            mask[:, t] = 1.0
        return actions

    def _replay_batch(self, actions):
        """WITH-grad replay of a fixed action sequence: recompute logits
        batched per-timestep (same cost shape as sampling), and return
        log_pf [B], mean entropy [B], and the resulting terminal spins
        [B, D] for the reward. Splitting sampling (no-grad) from replay
        (grad) like this mirrors the original code's separate log_pf/
        entropy_bonus passes, but each is now ONE batched pass over all D
        steps instead of D*B individual calls."""
        B, D = actions.shape
        spins = torch.zeros(B, D, device=self.device)
        mask = torch.zeros(B, D, device=self.device)
        log_pf = torch.zeros(B, device=self.device)
        
        ent = torch.zeros(B, device=self.device)
        for t in range(D):
            logits = self.policy(spins, mask, t)
            #logp = torch.log_softmax(logits, dim=-1)

            tilt_strength = getattr(self, "tilt_strength", 1.0)#tilt_strength = 1.0  # Hyperparameter: 0 disables tilt.
            candidates = spins[:, None, :].expand(-1, 2, -1).clone()
            candidates[:, :, t] = spins.new_tensor([-1.0, 1.0])
            energy_after = self.env.energy_batch(candidates.reshape(-1, self.env.D)).reshape(-1, 2)
            energy_before = self.env.energy_batch(spins).unsqueeze(1)
            transition_scores = -(energy_after - energy_before) / self.T

            logp = torch.log_softmax(logits + tilt_strength * transition_scores, dim=-1)
            a = actions[:, t]
            log_pf = log_pf + logp.gather(1, a.unsqueeze(1)).squeeze(1)
            p = logp.exp()
            ent = ent + (-(p * logp).sum(-1))
            spins[:, t] = torch.where(a == 0, -1.0, 1.0)
            mask[:, t] = 1.0
        return log_pf, ent / D, spins

    @torch.no_grad()
    def sample(self, n=500):
        """
        Returns a list of n NumPy arrays, each shape (D,), containing -1/+1. Converting to an array gives (n,25) for a 5×5 grid.
        """
        actions = self._sample_actions_batch(n, eps=0.0)
        _, _, spins = self._replay_batch(actions)
        return [spins[i].cpu().numpy() for i in range(n)] # list of n NumPy arrays of shape (25,) (or (D,))

    def loss(self, it, n_iter, batch_size):
        raise NotImplementedError  # TB and KC differ only here

    def train(self, n_iter, batch_size_fn=None, stop_check=None,
              loss_log=None, log_every=25):
        """batch_size_fn(it, n_iter) -> int, if given, lets the batch size
        change over training (e.g. ramp up) -- see make_batch_size_schedule().
        stop_check, if given, is a zero-arg callable checked before every
        iteration; if it returns True, training stops immediately and this
        returns False (instead of the usual True) to signal an INCOMPLETE
        run. Used for graceful, prompt exit on SLURM preemption -- see the
        checkpointing logic in run_tc_sweep.py, which discards (rather than
        keeps) a T-point interrupted mid-training, so no partially-trained
        result is ever silently treated as final.

        loss_log: if given, a list that (iteration, loss_value) tuples are
        appended to every `log_every` iterations, plus iteration 1 and the
        final iteration always -- lets a caller record a loss-vs-iteration
        curve without logging every single step (which would bloat saved
        output for long runs). Left as None (no logging, no overhead) by
        default -- pass a list explicitly to opt in."""
        for it in range(1, n_iter + 1):
            if stop_check is not None and stop_check():
                return False
            B = batch_size_fn(it, n_iter) if batch_size_fn is not None else BATCH
            loss = self.loss(it, n_iter, B)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.opt.step()
            if loss_log is not None and (it == 1 or it == n_iter or it % log_every == 0):
                loss_log.append((it, float(loss.item())))
        return True

class TBGFlowNet(BatchedGFlowNetBase):
    """
    Trajectory Balance: fits an explicit log_Z parameter so that
    log_Z + log_pf(tau) matches beta*log_reward(tau) for every tau.

    log_Z gets its OWN Adam param group at a much higher learning rate
    than the policy (lr_logz_mult x LR_NN, default 50x). Rationale: log_Z
    is a single scalar that has to track an EXTENSIVE quantity -- e.g.
    O(D) for SK at large D -- while Adam's per-parameter step size is
    roughly bounded by its lr regardless of gradient magnitude. At the
    policy's lr (tuned for a many-parameter network), log_Z simply cannot
    move far enough over a realistic number of iterations to track a
    large-D partition function; it isn't that TB is fundamentally unable
    to fit log_Z at scale, it's that it was starved of the step size to
    do so. This matters especially here because the paper's central claim
    is that KC's advantage comes from *not needing* log_Z -- so TB's
    log_Z needs a fair chance to converge, or a low-lr collapse gets
    mistaken for a fundamental scaling failure.

    log_Z is also warm-startable now (init_log_z), independent of the
    policy's own warm start, for the same reason the policy is warm-
    started: retraining log_Z from the fixed init D*log(2) at every T in
    a sweep -- even when the policy itself carries over -- was a second,
    compounding way TB was effectively handicapped relative to KC (which
    has no log_Z to reset).
    """
    def __init__(self, env, T, lr_logz_mult=LR_LOGZ_MULT, init_log_z=None, **kwargs):
        super().__init__(env, T, **kwargs)
        init_val = float(init_log_z) if init_log_z is not None else float(env.D * np.log(env.n_actions))
        self.log_Z = nn.Parameter(torch.tensor(init_val, device=self.device)) # TB learns log_Z as one scalar parameter
        self.opt = optim.Adam([
            {"params": self.policy.parameters(), "lr": LR_NN},
            {"params": [self.log_Z], "lr": LR_NN * lr_logz_mult}, # Adam optimizer assigns log Z its own learning rate, larger by factor of 50 in this case.
        ])

    def loss(self, it, n_iter, batch_size):
        beta, eps, ent_coef = self.schedule(it, n_iter)
        actions = self._sample_actions_batch(batch_size, eps=eps)
        log_pf, ent, spins = self._replay_batch(actions)
        logR = beta * (-self.env.energy_batch(spins) / self.T) # The log Reward is -\beta * E / T. This means higher temperature changes the ratios of different rewards! Higher T makes the ratio of two rewards, R_A/R_B higher. 
        residual = self.log_Z + log_pf - logR - LOG_PB_FIXED
        return (residual ** 2).mean() - ent_coef * ent.mean()

class KCGFlowNet(BatchedGFlowNetBase):
    """
    Kolmogorov-Criterion GFlowNet: no log_Z parameter. The per-trajectory
    residual delta(tau) = log_pf(tau) - beta*log_reward(tau) should be
    the SAME constant across trajectories at convergence (that constant
    is log_Z, but it cancels via telescoping and is never fit
    explicitly) -- so the loss is just the spread of that residual
    across the batch.

    Uses the FULL BATCH variance rather than disjoint pairs: Var[delta]
    is proportional to the average over ALL pairwise squared
    differences, so it uses every sample in the batch instead of only
    B/2 arbitrary pairings, for a lower-variance loss estimate at the
    same batch size (this was flagged as an improvement over the
    original disjoint-pair implementation and is essentially free to
    adopt from the start here).
    """
    def __init__(self, env, T, **kwargs):
        super().__init__(env, T, **kwargs)
        self.opt = optim.Adam(self.policy.parameters(), lr=LR_NN)

    def loss(self, it, n_iter, batch_size):
        beta, eps, ent_coef = self.schedule(it, n_iter)
        actions = self._sample_actions_batch(batch_size, eps=eps)
        log_pf, ent, spins = self._replay_batch(actions)
        logR = beta * (-self.env.energy_batch(spins) / self.T)
        residual = log_pf - logR - LOG_PB_FIXED
        return residual.var(unbiased=False) - ent_coef * ent.mean()

