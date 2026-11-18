"""
loop_gfn.py
===========
Loop-family, log Z-free GFlowNet objectives on top of spin_glass_lib.

Every objective here is a pairwise comparison of per-trajectory residuals
    V(tau) = log P_F(tau) - log P_B(tau | x) - beta * log R(x)
which must be the same constant (= log Z) for every complete trajectory at
the optimum. Pairing two trajectories closes a loop through the sink (or,
for small diamonds, inside the DAG), and the loss is 1/2 * (V - V')^2.
The 1/2 makes every term on the same scale as KC's batch variance, since
Var[V] = 1/2 * E[(V_i - V_j)^2] over independent pairs.

Three loop shapes:

  LARGE DIAMOND (w_ld)  two independent trajectories, full batch variance.
                        Identical to KCGFlowNet's loss on the tree DAG.
  KITE (w_kite)         a partner shares the base trajectory's first t_b
                        actions, takes a DIFFERENT action at step t_b, then
                        rolls out independently to the sink. The shared tail
                        cancels exactly, so only the post-branch segments
                        enter the loss (and only they get gradients).
                          t_b = 0     -> a forced-divergence large diamond
                          t_b = D-1   -> one-step kite: the local-field
                                         constraint log P_F(+|s) - log P_F(-|s)
                                         = -beta * dE_flip / T
  SMALL DIAMOND (w_sd)  free-order DAG only. Swap two consecutive actions
                        (a, b) -> (b, a) of the base trajectory; both paths
                        connect the same two states. Reward cancels, and with
                        uniform P_B so does P_B. The loss is purely
                        order-consistency of P_F:
                          [lp(a|s) + lp(b|s+a)] - [lp(b|s) + lp(a|s+b)]

DAG toggle (dag=...):
  "tree"  fixed site order 0..D-1, action = value (2 actions). P_B = 1.
          This is the existing KC/TB setup.
  "free"  any unassigned site may be set next, action = 2*site + value.
          P_B is uniform over assigned sites. Every complete trajectory then
          has log P_B(tau|x) = -log D!, and every post-branch segment of a
          given length has the same P_B product, so P_B cancels from EVERY
          loss below and no P_B network is needed.

TB (w_tb) is also available on both DAGs as a baseline with an explicit
log Z parameter (own Adam group at LR_LOGZ_MULT x lr, as in TBGFlowNet):
    1/B sum (log Z + log P_F(tau) - log P_B(tau|x) - beta * log R(x))^2

Cycle-basis note: on the free DAG, small diamonds (squares) plus one-step
kites generate the full cycle space including the sink. On the tree there
are no squares, and one-step kites only relate last-level siblings, so
kites at all depths are needed there.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.gflownet import (BatchedGFlowNetBase, StepEmbedPolicy, LR_NN, LR_LOGZ_MULT,)

# Preset -> (w_ld, w_kite, w_sd, kite_lo, kite_hi)
LOOP_PRESETS = {
    "ld":        (1.0, 0.0, 0.0, 0.0, 1.0),   # == KC
    "kite":      (0.0, 1.0, 0.0, 0.0, 1.0),   # kites, uniform branch depth
    "kite_last": (0.0, 1.0, 0.0, 1.0, 1.0),   # one-step kites only
    "ld_kite":   (1.0, 1.0, 0.0, 0.0, 1.0),
    "ld_sd":     (1.0, 0.0, 1.0, 0.0, 1.0),   # free DAG only
    "kite_sd":   (0.0, 1.0, 1.0, 0.0, 1.0),   # free DAG only
    "basis":     (0.0, 1.0, 1.0, 1.0, 1.0),   # free DAG only: squares + one-step kites
}


class FreeOrderPolicy(nn.Module):
    """
    Policy for the free-order DAG. Same MLP trunk as StepEmbedPolicy, but
    the head emits 2*D logits (site, value) instead of 2. Action index
    a = 2*site + value (value 0 -> -1, 1 -> +1), so a ^ 1 flips the value
    at the same site, matching the tree's a ^ 1 convention.

    step may be a python int (whole batch in lockstep) or a [B] long tensor
    (needed for small diamonds, where rows sit at different depths). The
    step is always mask.sum(), so it adds no information beyond the mask;
    it's kept for parity with the tree policy.
    """
    def __init__(self, D, hidden=128, step_embed_dim=16):
        super().__init__()
        self.D = D
        self.step_embed = nn.Embedding(D + 1, step_embed_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * D + step_embed_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 2 * D),
        )

    def forward(self, spins, mask, step):
        B = spins.shape[0]
        if not torch.is_tensor(step):
            step = torch.full((B,), int(step), dtype=torch.long, device=spins.device)
        feats = torch.cat([spins, mask, self.step_embed(step)], dim=-1)
        return self.net(feats)  # [B, 2D], legality masking done by the caller


class LoopGFlowNet(BatchedGFlowNetBase):
    def __init__(self, env, T, dag="tree", w_ld=1.0, w_kite=0.0, w_sd=0.0,
                 kite_lo=0.0, kite_hi=1.0, n_kite=1, n_sd=1, kite_diverge="flip",
                 sd_all=False, kite_all=False, w_tb=0.0, **kwargs):
        if dag not in ("tree", "free"):
            raise ValueError(f"dag must be 'tree' or 'free', got {dag!r}")
        if w_sd > 0 and dag == "tree":
            raise ValueError("small diamonds need dag='free' (a tree has no squares)")
        if kite_diverge not in ("flip", "any"):
            raise ValueError("kite_diverge must be 'flip' or 'any'")
        if not (0.0 <= kite_lo <= kite_hi <= 1.0):
            raise ValueError("need 0 <= kite_lo <= kite_hi <= 1")
        init_sd = kwargs.pop("init_state_dict", None)
        super().__init__(env, T, init_state_dict=None, **kwargs)
        self.dag = dag
        if dag == "free":
            self.policy = FreeOrderPolicy(env.D).to(self.device)
        if init_sd is not None:
            self.policy.load_state_dict(init_sd)
        self.w_ld, self.w_kite, self.w_sd = float(w_ld), float(w_kite), float(w_sd)
        self.kite_lo, self.kite_hi = float(kite_lo), float(kite_hi)
        self.n_kite, self.n_sd = int(n_kite), int(n_sd)
        self.kite_diverge = kite_diverge
        # dense basis sampling (free DAG): every small diamond along each base
        # trajectory, and a one-step kite at EVERY site of each terminal x
        # (any site can be the last one set), instead of one of each per trajectory
        self.sd_all, self.kite_all = bool(sd_all), bool(kite_all)
        if (sd_all or kite_all) and dag != "free":
            raise ValueError("sd_all / kite_all need dag='free'")
        self.n_act = 2 if dag == "tree" else 2 * env.D
        self.w_tb = float(w_tb)
        # free DAG: log P_B(tau|x) = -log D! for every trajectory
        self.log_pb = -math.lgamma(env.D + 1) if dag == "free" else 0.0
        groups = [{"params": self.policy.parameters(), "lr": LR_NN}]
        if self.w_tb > 0:
            # same init as TBGFlowNet: log Z of the uniform policy's flow, D * log 2
            self.log_Z = nn.Parameter(torch.tensor(env.D * math.log(2.0), device=self.device))
            groups.append({"params": [self.log_Z], "lr": LR_NN * LR_LOGZ_MULT})
        self.opt = optim.Adam(groups)
        self.last_components = {}

    # ------------------------------------------------------------
    #  DAG primitives
    # ------------------------------------------------------------
    def _logp(self, spins, mask, t):
        """Log-probs over the action space at the given states. Illegal
        actions (free DAG: already-assigned sites) get -inf."""
        logits = self._policy_fwd(spins, mask, t)
        if self.dag == "free":
            illegal = mask.repeat_interleave(2, dim=1) > 0.5
            logits = logits.masked_fill(illegal, float("-inf"))
        return torch.log_softmax(logits, dim=-1)

    def _legal(self, mask):
        if self.dag == "tree":
            return torch.ones(mask.shape[0], 2, device=mask.device)
        return (mask.repeat_interleave(2, dim=1) < 0.5).float()

    def _apply(self, spins, mask, a, t, rows=None):
        """In-place: apply actions a [M] at step t (rows: optional bool
        selector; unselected rows untouched)."""
        if self.dag == "tree":
            site = torch.full_like(a, t)
        else:
            site = a // 2
        val = torch.where(a % 2 == 0, -1.0, 1.0)
        if rows is None:
            rows = torch.ones_like(a, dtype=torch.bool)
        r = rows.nonzero(as_tuple=True)[0]
        spins[r, site[r]] = val[r]
        mask[r, site[r]] = 1.0

    def _prefix_state(self, acts, t):
        """States after applying acts[:, :t[i]] for each row (no policy
        calls). acts [M, D], t [M] long. Returns spins, mask [M, D]."""
        M, D = acts.shape
        steps = torch.arange(D, device=acts.device).unsqueeze(0)
        keep = (steps < t.unsqueeze(1)).to(torch.get_default_dtype())
        if self.dag == "tree":
            site = steps.expand(M, D)
        else:
            site = acts // 2
        val = torch.where(acts % 2 == 0, -1.0, 1.0)
        spins = torch.zeros(M, D, device=acts.device).scatter_(1, site, val * keep)
        mask = torch.zeros(M, D, device=acts.device).scatter_(1, site, keep)
        return spins, mask

    # ------------------------------------------------------------
    #  Rollout (no grad) and replay (with grad)
    # ------------------------------------------------------------
    @torch.no_grad()
    def _rollout(self, M, eps, base=None, tb=None):
        """Sample M trajectories from q = (1-eps)*pi + eps*uniform(legal).
        If base [M, D] and tb [M] are given, copy base for steps < tb, take
        a forced-different action at step tb, and sample freely after.
        Returns acts [M, D]."""
        D = self.env.D
        acts = torch.zeros(M, D, dtype=torch.long, device=self.device)
        if base is None:
            t0 = 0
            spins = torch.zeros(M, D, device=self.device)
            mask = torch.zeros(M, D, device=self.device)
        else:
            t0 = int(tb.min().item())
            acts[:, :t0] = base[:, :t0]
            spins, mask = self._prefix_state(base, torch.full_like(tb, t0))
        for t in range(t0, D):
            logp = self._logp(spins, mask, t)
            legal = self._legal(mask)
            q = (1.0 - eps) * logp.exp() + eps * legal / legal.sum(-1, keepdim=True)
            if base is not None:
                copy = tb > t
                branch = tb == t
                if branch.any():
                    if self.kite_diverge == "flip":
                        q_b = torch.zeros_like(q)
                        q_b.scatter_(1, (base[:, t] ^ 1).unsqueeze(1), 1.0)
                    else:
                        q_b = q.clone()
                        q_b.scatter_(1, base[:, t].unsqueeze(1), 0.0)
                    q = torch.where(branch.unsqueeze(1), q_b, q)
            a = torch.multinomial(q, 1).squeeze(1)
            if base is not None:
                a = torch.where(copy, base[:, t], a)
            acts[:, t] = a
            self._apply(spins, mask, a, t)
        return acts

    def _replay(self, acts, t_from=None, want_entropy=False):
        """WITH-grad replay. Returns per-step log P_F [M, D] (zeros for
        steps < t_from, which are never evaluated), per-step entropy sum
        [M] (or None), and terminal spins [M, D]."""
        M, D = acts.shape
        if t_from is None:
            t0 = 0
            spins = torch.zeros(M, D, device=self.device)
            mask = torch.zeros(M, D, device=self.device)
        else:
            t0 = int(t_from.min().item())
            spins, mask = self._prefix_state(acts, torch.full_like(t_from, t0))
        cols = [torch.zeros(M, device=self.device)] * t0
        ent = torch.zeros(M, device=self.device) if want_entropy else None
        for t in range(t0, D):
            logp = self._logp(spins, mask, t)
            a = acts[:, t]
            cols.append(logp.gather(1, a.unsqueeze(1)).squeeze(1))
            if want_entropy:
                # zero out -inf (illegal) entries before the product: 0 * -inf
                # inside torch.where still poisons the backward pass with NaN
                safe = logp.masked_fill(~torch.isfinite(logp), 0.0)
                ent = ent - (logp.exp() * safe).sum(-1)
            spins = spins.clone(); mask = mask.clone()
            self._apply(spins, mask, a, t)
        return torch.stack(cols, dim=1), ent, spins

    @torch.no_grad()
    def sample(self, n=500):
        acts = self._rollout(n, eps=0.0)
        _, _, spins = self._replay(acts)
        return [spins[i].cpu().numpy() for i in range(n)]

    # ------------------------------------------------------------
    #  Loss
    # ------------------------------------------------------------
    def _logR(self, spins, beta):
        return beta * (-self.env.energy_batch(spins) / self.T)

    def _sample_tb(self, M):
        D = self.env.D
        lo = int(round(self.kite_lo * (D - 1)))
        hi = int(round(self.kite_hi * (D - 1)))
        return torch.randint(lo, hi + 1, (M,), device=self.device)

    def loss(self, it, n_iter, batch_size):
        beta, eps, ent_coef = self.schedule(it, n_iter)
        D = self.env.D
        B = batch_size
        base = self._rollout(B, eps)
        lp_base, ent, spins_base = self._replay(base, want_entropy=True)
        logR_base = self._logR(spins_base, beta)
        total = -ent_coef * (ent / D).mean()
        comps = {}

        if self.w_ld > 0:
            V = lp_base.sum(1) - logR_base
            l_ld = V.var(unbiased=False)
            total = total + self.w_ld * l_ld
            comps["ld"] = l_ld.item()

        if self.w_tb > 0:
            res = self.log_Z + lp_base.sum(1) - self.log_pb - logR_base
            l_tb = (res ** 2).mean()
            total = total + self.w_tb * l_tb
            comps["tb"] = l_tb.item()

        if self.w_kite > 0 and self.kite_all:
            # x with site i removed, for every i: [B*D] level-(D-1) states
            Bx = spins_base.shape[0]
            rows = torch.arange(Bx, device=self.device).repeat_interleave(D)
            site = torch.arange(D, device=self.device).repeat(Bx)
            sp = spins_base[rows].clone(); mk = torch.ones_like(sp)
            idx = torch.arange(len(rows), device=self.device)
            v = sp[idx, site]
            sp[idx, site] = 0.0; mk[idx, site] = 0.0
            lp_k = self._logp(sp, mk, D - 1)
            a = 2 * site + (v > 0).long()
            dlp = lp_k.gather(1, a[:, None]).squeeze(1) - lp_k.gather(1, (a ^ 1)[:, None]).squeeze(1)
            # logR(x) - logR(x with spin i flipped) = beta * (E(x^i) - E(x)) / T
            xf = spins_base[rows].clone(); xf[idx, site] = -v
            dlogR = logR_base[rows] - self._logR(xf, beta)
            l_kite = 0.5 * ((dlp - dlogR) ** 2).mean()
            total = total + self.w_kite * l_kite
            comps["kite"] = l_kite.item()
        elif self.w_kite > 0:
            K = self.n_kite
            base_rep = base.repeat(K, 1)
            tb = self._sample_tb(B * K)
            partner = self._rollout(B * K, eps, base=base_rep, tb=tb)
            lp_p, _, spins_p = self._replay(partner, t_from=tb)
            seg = (torch.arange(D, device=self.device).unsqueeze(0) >= tb.unsqueeze(1)).float()
            V_b = (lp_base.repeat(K, 1) * seg).sum(1) - logR_base.repeat(K)
            V_p = (lp_p * seg).sum(1) - self._logR(spins_p, beta)
            l_kite = 0.5 * ((V_b - V_p) ** 2).mean()
            total = total + self.w_kite * l_kite
            comps["kite"] = l_kite.item()

        if self.w_sd > 0:
            K = self.n_sd
            if self.sd_all:
                rows = torch.arange(B, device=self.device).repeat_interleave(D - 1)
                t = torch.arange(D - 1, device=self.device).repeat(B)
            elif 1 < K <= D - 1:
                # K distinct positions per base trajectory (no repeated diamonds)
                rows = torch.arange(B, device=self.device).repeat(K)
                t = torch.rand(B, D - 1, device=self.device).argsort(1)[:, :K].T.reshape(-1)
            else:
                rows = torch.arange(B, device=self.device).repeat(K)
                t = torch.randint(0, D - 1, (B * K,), device=self.device)
            acts = base[rows]
            a = acts.gather(1, t.unsqueeze(1)).squeeze(1)
            b = acts.gather(1, (t + 1).unsqueeze(1)).squeeze(1)
            orig = lp_base[rows, t] + lp_base[rows, t + 1]
            spins_s, mask_s = self._prefix_state(acts, t)
            lp_s = self._logp(spins_s, mask_s, t)
            sp2, mk2 = spins_s.clone(), mask_s.clone()
            sb = b // 2
            idx = torch.arange(len(rows), device=self.device)
            sp2[idx, sb] = torch.where(b % 2 == 0, -1.0, 1.0)
            mk2[idx, sb] = 1.0
            lp_sb = self._logp(sp2, mk2, t + 1)
            swap = lp_s.gather(1, b.unsqueeze(1)).squeeze(1) + lp_sb.gather(1, a.unsqueeze(1)).squeeze(1)
            l_sd = 0.5 * ((orig - swap) ** 2).mean()
            total = total + self.w_sd * l_sd
            comps["sd"] = l_sd.item()

        self.last_components = comps
        return total
