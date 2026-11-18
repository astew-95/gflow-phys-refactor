"""
kc_exact.py
===========
Exact, tabular, noise-free test bed for the Kolmogorov-criterion (KC) loop
objectives on small SK instances. Settles whether the basis objective's
failure (HANDOFF step 4b) is a property of the OBJECTIVE or of the practical
setup (sampling noise, network capacity, learning-rate schedule).

Every partial state of the free-order DAG is enumerated (3^D states, digit
0 = unassigned, 1 = spin -1, 2 = spin +1). The policy is a lookup table of
logits theta[s, a], a = 2*site + value, so there is no function approximation.
Every loss is an EXACT expectation over the sampling distribution (no
sampled minibatches), with the same gradient convention as training: the
sampling weights are held fixed (detached) and gradients flow only through
the log-probabilities inside the residuals. Same weights as loop_gfn.py:

  ld     Var_tau[V],  V = log P_F(tau) - log P_B(tau|x) - log R(x)
  kite   1/2 E[r_k^2] over one-step kites at the last step
  sd     1/2 E[r_sd^2], depth t uniform on 0..D-2 (one diamond per trajectory)
  basis  kite + sd
  ld_sd  ld + sd   (large + small diamonds)
  tb     E[(log Z + V)^2] with a learned scalar log Z (TB baseline; = ld at the best log Z)

dag="tree" reuses the same tables with only site k legal at level k (fixed
order, P_B = 1, no squares). kind="ising" swaps SK for the N x N periodic
Ising model (D = N^2; J is then only the init/sampling seed).

Loss weightings (which states/edges the residuals are averaged over):
  onpolicy   forward rollouts of the current policy (as in training)
  uniform    "perfect coverage": every state of a level equally weighted
  target     REPLAY BUFFER, ideal: x ~ p, replayed BACKWARD under uniform P_B.
             Equals on-policy weighting under the optimal policy.
  selfbuf    REPLAY BUFFER, realistic: x ~ q (the policy's own terminal samples),
             replayed backward. Differs from onpolicy exactly when P_F is
             order-inconsistent (then backward paths to x are not q's paths).
  mix_target / mix_selfbuf   50/50 mixture with onpolicy (half batch on-policy,
             half from the buffer, the usual GFlowNet replay setup)

Subcommands:
  python kc_exact.py perturb --D 8 --T 0.4 --J 0             # conditioning test
  python kc_exact.py train --D 8 --T 0.4 --J 0 --method basis --lr 0.1 --sched cos
  python kc_exact.py grid-list / grid-run --task i           # SLURM array driver
"""
import argparse, itertools, json, math, os, time
import numpy as np
import torch

from physics.spin_system import SpinEnv
from physics.spin_glass_lib import LR_LOGZ_MULT

torch.set_default_dtype(torch.float64)


class ExactDAG:
    def __init__(self, D, T, J_seed=0, dag="free", kind="sk"):
        self.D, self.T, self.dag, self.kind = D, T, dag, kind
        N = 3 ** D
        codes = torch.arange(N)
        pw = 3 ** torch.arange(D)
        digits = (codes.unsqueeze(1) // pw) % 3                       # [N, D]
        self.level = (digits > 0).sum(1)
        A = 2 * D
        site = torch.arange(A) // 2
        val = torch.arange(A) % 2
        legal = digits[:, site] == 0                                  # [N, A]
        if dag == "tree":
            legal &= site.unsqueeze(0) == self.level.unsqueeze(1)
        self.legal = legal
        self.child = torch.where(legal, codes.unsqueeze(1) + (1 + val) * pw[site], 0)
        self.by_level = [torch.nonzero(self.level == k).squeeze(1) for k in range(D + 1)]
        env = make_env(kind, D, J_seed)
        term = self.by_level[D]
        X = (2 * digits[term] - 3).to(torch.get_default_dtype())     # 1 -> -1, 2 -> +1
        self.logR = torch.zeros(N)
        self.logR[term] = -env.energy_batch(X) / T
        self.logZ = float(torch.logsumexp(self.logR[term], 0))
        self.log_pb = -math.lgamma(D + 1) if dag == "free" else 0.0  # log P_B(tau|x), constant

    # -------------------------------------------------------------- policy
    def logp(self, theta):
        lp = torch.log_softmax(theta.masked_fill(~self.legal, float("-inf")), dim=1)
        return lp.masked_fill(~self.legal, 0.0)                       # safe zeros, always masked later

    def optimal_theta(self):
        """Exact log P_F for this DAG (free: uniform P_B; tree: P_B = 1)."""
        D = self.D
        lz = self.logR.clone()
        pw = 3 ** torch.arange(D)
        for k in range(D - 1, -1, -1):
            s = self.by_level[k]
            i = torch.argmax(self.legal[s].view(len(s), D, 2)[:, :, 0].to(torch.int8), 1)  # first legal site
            lz[s] = torch.logaddexp(lz[s + pw[i]], lz[s + 2 * pw[i]])
        if self.dag == "free":
            lf = lz - torch.tensor([math.log(math.comb(D, int(k))) for k in range(D + 1)])[self.level]
            th = lf[self.child] - lf.unsqueeze(1) - torch.log(self.level.unsqueeze(1) + 1.0)
        else:
            th = lz[self.child] - lz.unsqueeze(1)
        return th.masked_fill(~self.legal, 0.0)

    # -------------------------------------------------------------- exact sampling weights
    @torch.no_grad()
    def reach(self, pi):
        """rho[s] = P(trajectory visits s) under the transition table pi [N, A]."""
        rho = torch.zeros(len(pi)); rho[0] = 1.0
        for k in range(self.D):
            s = self.by_level[k]
            rho.index_add_(0, self.child[s].flatten(), (rho[s].unsqueeze(1) * pi[s]).flatten())
        return rho

    @torch.no_grad()
    def uniform_weights(self):
        """'Perfect coverage': every state of a level and every legal action equally weighted."""
        pi = self.legal.to(torch.get_default_dtype())
        pi = pi / pi.sum(1, keepdim=True).clamp_min(1)
        rho = torch.zeros(len(pi))
        for k in range(self.D + 1):
            rho[self.by_level[k]] = 1.0 / len(self.by_level[k])
        return pi, rho

    @torch.no_grad()
    def backward_weights(self, qx):
        """Replay-buffer weights: terminal x ~ qx (over by_level[D]), trajectories
        sampled BACKWARD with uniform P_B. Returns forward-view (pi_w, rho): rho[s] =
        P(backward path visits s), pi_w[s, a] = P(edge s -> s+a | s) under that process."""
        rho = torch.zeros(len(self.legal)); rho[self.by_level[self.D]] = qx
        pi = torch.zeros(self.legal.shape)
        for k in range(self.D - 1, -1, -1):
            s = self.by_level[k]
            pb = 1.0 / (k + 1) if self.dag == "free" else 1.0               # P_B(s | child), child at level k+1
            w = torch.where(self.legal[s], rho[self.child[s]] * pb, 0.0)  # flow on edge s -> s+a
            rho[s] = w.sum(1)
            pi[s] = w / rho[s].clamp_min(1e-300).unsqueeze(1)
        return pi, rho

    @torch.no_grad()
    def target_q(self):
        return (self.logR[self.by_level[self.D]] - self.logZ).exp()

    # -------------------------------------------------------------- metrics
    @torch.no_grad()
    def metrics(self, theta):
        pi = self.logp(theta).exp() * self.legal
        q = self.reach(pi)[self.by_level[self.D]]
        lq = torch.log(q.clamp_min(1e-300))
        lp = self.logR[self.by_level[self.D]] - self.logZ
        p = lp.exp()
        return dict(kl_qp=float((q * (lq - lp)).sum()), kl_pq=float((p * (lp - lq)).sum()),
                    tv=float(0.5 * (q - p).abs().sum()))

    # -------------------------------------------------------------- losses
    def losses(self, theta, which=("ld", "kite", "sd"), weights="onpolicy", logZ=None):
        lp = self.logp(theta)
        if weights.startswith("mix_"):                                 # half on-policy, half buffer
            a = self._losses(lp, which, *self.weights(lp, "onpolicy"), logZ)
            b = self._losses(lp, which, *self.weights(lp, weights[4:]), logZ)
            return {k: 0.5 * (a[k] + b[k]) for k in a}
        return self._losses(lp, which, *self.weights(lp, weights), logZ)

    @torch.no_grad()
    def weights(self, lp, weights):
        if weights == "onpolicy":
            pi_w = lp.exp() * self.legal
            return pi_w, self.reach(pi_w)
        if weights == "uniform":
            return self.uniform_weights()
        if weights == "target":
            return self.backward_weights(self.target_q())
        if weights == "selfbuf":
            q = self.reach(lp.exp() * self.legal)[self.by_level[self.D]]
            return self.backward_weights(q)
        raise ValueError(weights)

    def _losses(self, lp, which, pi_w, rho, logZ=None):
        D = self.D
        out = {}
        if "ld" in which or "tb" in which:
            # first/second moments of the path sum of log P_F, pushed level by level
            Am = torch.zeros(len(lp)); Bm = torch.zeros(len(lp))
            for k in range(D):
                s = self.by_level[k]
                w, l = pi_w[s], lp[s]
                ch = self.child[s].flatten()
                a, b = Am[s].unsqueeze(1), Bm[s].unsqueeze(1)
                r = rho[s].unsqueeze(1)
                Am = Am.index_add(0, ch, (w * (a + r * l)).flatten())
                Bm = Bm.index_add(0, ch, (w * (b + 2 * l * a + r * l * l)).flatten())
            t = self.by_level[D]
            c = -self.log_pb - self.logR[t]
            EV = (Am[t] + rho[t] * c).sum()
            EV2 = (Bm[t] + 2 * c * Am[t] + rho[t] * c * c).sum()
            if "ld" in which:
                out["ld"] = EV2 - EV * EV
            if "tb" in which:                                         # E[(logZ + V)^2]
                out["tb"] = EV2 + 2 * logZ * EV + logZ * logZ
        if "kite" in which:
            s = self.by_level[D - 1]
            l, ch = lp[s], self.child[s]
            flip = torch.arange(2 * D) ^ 1
            r = (l - l[:, flip]) - (self.logR[ch] - self.logR[ch[:, flip]])
            w = rho[s].unsqueeze(1) * pi_w[s] * self.legal[s]
            out["kite"] = 0.5 * (w * r * r).sum() / w.sum()
        if "sd" in which:
            tot = 0.0
            for k in range(D - 1):
                s = self.by_level[k]
                l, ch = lp[s], self.child[s]                             # [n, A]
                lch = lp[ch]                                             # [n, A(a), A(b)]: log pi(b | s+a)
                r = (l.unsqueeze(2) + lch) - (l.unsqueeze(1) + lch.transpose(1, 2))
                ok = self.legal[s].unsqueeze(2) & self.legal[ch]                 # a at s, b at s+a
                ok = ok & ok.transpose(1, 2)                                     # b at s, a at s+b
                w = rho[s][:, None, None] * pi_w[s].unsqueeze(2) * pi_w[ch] * ok
                tot = tot + 0.5 * (w * r * r).sum() / (D - 1)
            out["sd"] = tot
        return out


def make_env(kind, D, J_seed):
    if kind == "ising":
        N = int(round(math.sqrt(D))); assert N * N == D
        return SpinEnv("ising", N=N)
    return SpinEnv("sk", D=D, p_af=0.5, J_seed=J_seed)


def method_weights(method):
    """(w_ld, w_kite, w_sd, w_tb). Weighted variants: basis_kw<w> = w*kite + sd (upweighted
    terminal kites), ld_sdw<w> = ld + w*sd, ld_basis = ld + kite + sd."""
    m = method.removesuffix("_dense")
    fixed = dict(ld=(1, 0, 0, 0), tb=(0, 0, 0, 1), ld_sd=(1, 0, 1, 0), basis=(0, 1, 1, 0),
                 ld_basis=(1, 1, 1, 0))
    if m in fixed:
        return tuple(float(x) for x in fixed[m])
    if m.startswith("basis_kw"):
        return (0.0, float(m[8:]), 1.0, 0.0)
    if m.startswith("ld_sdw"):
        return (1.0, 0.0, float(m[6:]), 0.0)
    raise ValueError(method)


def objective(L, method):
    w = dict(zip(("ld", "kite", "sd", "tb"), method_weights(method)))
    return sum(w[k] * L[k] for k in L)


def which_of(method):
    return tuple(k for k, w in zip(("ld", "kite", "sd", "tb"), method_weights(method)) if w > 0)


# ------------------------------------------------------------------ experiments
def cmd_perturb(a):
    """Conditioning: at the exact optimum + noise, how big is KL for a given loss?"""
    g = ExactDAG(a.D, a.T, a.J, "free")
    th0 = g.optimal_theta()
    rows = []
    gen = torch.Generator().manual_seed(a.seed)
    for kind in ("white", "smooth"):
        if kind == "white":
            xi = torch.randn(th0.shape, generator=gen)
        else:   # same perturbation at every state of a level: systematic, low-frequency
            xi = torch.randn(a.D + 1, 2 * a.D, generator=gen)[g.level]
        for sig in a.sigmas:
            th = th0 + sig * xi * g.legal
            L = {k: float(v) for k, v in g.losses(th).items()}
            rows.append(dict(kind=kind, sigma=sig, **L, basis=L["kite"] + L["sd"], **g.metrics(th)))
    res = dict(D=a.D, T=a.T, J=a.J, rows=rows)
    print(json.dumps(res))
    return res


class TablePolicy(torch.nn.Module):
    """Lookup-table policy usable by LoopGFlowNet: (spins, mask, step) -> theta[state code]."""
    def __init__(self, D):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(3 ** D, 2 * D))
        self.register_buffer("pw", 3 ** torch.arange(D))

    def code(self, spins, mask):
        dig = (mask > 0.5).long() * (1 + (spins > 0).long())
        return (dig * self.pw).sum(1)

    def forward(self, spins, mask, step):
        return self.theta[self.code(spins, mask)]


def train(D, T, J, method, dag, lr, sched, iters, weights="onpolicy", policy="table",
          grad="exact", B=64, n_log=60, seed=0, kind="sk"):
    """policy: 'table' (no function approximation) or 'net' (FreeOrderPolicy MLP).
    grad: 'exact' (full expectation) or 'sampled' (the real LoopGFlowNet.loss with batch B,
    at beta = 1, eps = EPS_END, no entropy bonus)."""
    from experiments.scripts.loop_gfn import LoopGFlowNet, FreeOrderPolicy
    torch.manual_seed(seed); np.random.seed(seed)
    g = ExactDAG(D, T, J, dag, kind)
    pol = TablePolicy(D) if policy == "table" else FreeOrderPolicy(D, hidden=512 if policy == "net512" else 128)
    states = torch.arange(3 ** D)
    dig = (states.unsqueeze(1) // pol_pw(D)) % 3
    sp_all, mk_all = (dig == 2).double() - (dig == 1).double(), (dig > 0).double()
    theta_all = (lambda: pol.theta) if policy == "table" else (lambda: pol(sp_all, mk_all, g.level))
    if grad == "sampled":
        w = method_weights(method)
        lg = LoopGFlowNet(make_env(kind, D, J), T, dag=dag, w_ld=w[0],
                          w_kite=w[1], w_sd=w[2], w_tb=w[3], kite_lo=0.0 if method == "ld" else 1.0, kite_hi=1.0,
                          sd_all=method.endswith("_dense"), kite_all=method == "basis_dense")
        lg.policy = pol
    method = method.removesuffix("_dense")                           # exact logging objective
    # TB: scalar log Z at LR_LOGZ_MULT x lr (as TBGFlowNet), capped for the table's large lr
    logZ = (lg.log_Z if grad == "sampled" else torch.nn.Parameter(torch.tensor(D * math.log(2.0)))) \
        if method == "tb" else None
    groups = [dict(params=list(pol.parameters()), lr=lr, base=lr)]
    if logZ is not None:
        zlr = min(lr * LR_LOGZ_MULT, 0.1)
        groups.append(dict(params=[logZ], lr=zlr, base=zlr))
    opt = torch.optim.Adam(groups)
    which = which_of(method)
    log_at = set(np.unique(np.geomspace(1, iters, n_log).astype(int)).tolist()) | {iters}
    curve = []
    t0 = time.time()
    for it in range(1, iters + 1):
        if sched == "cos":
            for pg in opt.param_groups:
                pg["lr"] = pg["base"] * 0.5 * (1 + math.cos(math.pi * (it - 1) / iters))
        if grad == "exact":
            L = g.losses(theta_all(), which, weights, logZ)
            loss = objective(L, method)
        else:
            loss = lg.loss(iters, iters, B)
        opt.zero_grad(); loss.backward()
        if policy == "net":
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)   # as in practical training
        opt.step()
        if it in log_at:
            with torch.no_grad():
                th = theta_all().detach()
                Lx = {k: float(v) for k, v in g.losses(th, which, weights, logZ).items()}
                extra = dict(logZ=float(logZ), logZ_true=g.logZ) if logZ is not None else {}
                curve.append(dict(it=it, loss=float(objective(Lx, method)), **Lx, **extra, **g.metrics(th)))
    return dict(kind=kind, D=D, T=T, J=J, method=method, dag=dag, lr=lr, sched=sched, iters=iters,
                weights=weights, policy=policy, grad=grad, B=B, seed=seed,
                curve=curve, final=curve[-1], secs=time.time() - t0)


def pol_pw(D):
    return 3 ** torch.arange(D)


# Ladder grid: add the practical ingredients one at a time.
#   policy in {table, net}  x  grad in {exact, sampled}; basis also with uniform
#   ("perfect coverage") weights where the gradient is exact.
LADDER = []
for pol, grad, lrs, scheds in [("table", "exact", [0.1], ["cos"]),
                               ("table", "sampled", [0.03, 0.1], ["cos"]),
                               ("net", "exact", [1e-3, 3e-3], ["cos"]),
                               ("net", "sampled", [1e-3], ["const", "cos"])]:
    for m in ("ld", "basis"):
        for lr in lrs:
            for sc in scheds:
                LADDER.append(dict(policy=pol, grad=grad, method=m, weights="onpolicy", lr=lr, sched=sc))
    if grad == "exact":
        for lr in lrs:
            LADDER.append(dict(policy=pol, grad=grad, method="basis", weights="uniform", lr=lr, sched="cos"))
# round 2: dense basis sampling (B=64, and B=16 = compute-matched to ld) and a 4x wider net
LADDER2 = [dict(policy="net", grad="sampled", method="basis_dense", weights="onpolicy", lr=1e-3,
                sched="cos", B=B) for B in (16, 64)]
LADDER2 += [dict(policy="net512", grad="exact", method=m, weights=w, lr=lr, sched="cos")
            for m, w in (("ld", "onpolicy"), ("basis", "onpolicy"), ("basis", "uniform")) for lr in (1e-3, 3e-3)]
GRID_D = [6, 8, 10]
GRID_T = [1.0, 0.4]
GRID_J = [0, 1]
ITERS = 4000


# 3x3 periodic Ising (D = 9): does basis learn a trivial lattice case, and does a
# replay buffer (backward replay of terminal samples) fix it? J = init seed here.
BUFFERS = ("onpolicy", "uniform", "target", "selfbuf", "mix_target", "mix_selfbuf")
LADDER_ISING = []
for pol, grad, lrs in [("table", "exact", [0.1]), ("net", "exact", [1e-3, 3e-3])]:
    for lr in lrs:
        LADDER_ISING.append(dict(policy=pol, grad=grad, method="ld", weights="onpolicy", lr=lr, sched="cos"))
        LADDER_ISING += [dict(policy=pol, grad=grad, method="basis", weights=w, lr=lr, sched="cos") for w in BUFFERS]
        LADDER_ISING += [dict(policy=pol, grad=grad, method=m, weights="onpolicy", lr=lr, sched="cos", dag="tree")
                         for m in ("ld", "basis")]                   # tree: basis = one-step kites only
LADDER_ISING += [dict(policy="net", grad="sampled", method=m, weights="onpolicy", lr=1e-3, sched="cos")
                 for m in ("ld", "basis", "basis_dense")]
GRID_ISING_T = [2.27, 1.5]

# 3x3 Ising: large + small diamonds together (ld_sd) vs TB vs ld alone. Free DAG
# (squares need it); tree tb/ld as exact-gradient references. No replay buffer.
LADDER_LDSD = []
for pol, grad, lrs in [("table", "exact", [0.1]), ("net", "exact", [1e-3, 3e-3])]:
    for lr in lrs:
        LADDER_LDSD += [dict(policy=pol, grad=grad, method=m, weights="onpolicy", lr=lr, sched="cos")
                        for m in ("tb", "ld", "ld_sd")]
        LADDER_LDSD += [dict(policy=pol, grad=grad, method=m, weights="onpolicy", lr=lr, sched="cos", dag="tree")
                        for m in ("tb", "ld")]
LADDER_LDSD += [dict(policy="net", grad="sampled", method=m, weights="onpolicy", lr=1e-3, sched="cos")
                for m in ("tb", "ld", "ld_sd", "ld_sd_dense")]
GRID_LDSD_SEEDS = [0, 1, 2]

# Advisor's suggestions (HANDOFF 4g): upweight the terminal one-step kites in basis
# (basis_kw<w>), give small diamonds real weight next to ld (ld_sdw<w>), and combine
# full paths with the local loops (ld_basis). Run where basis already fails: net with
# exact gradients (lr 1e-3/3e-3) and net with sampled gradients. ld/basis references
# have the same task names as the sk ladder / ising grids and are linked in, not rerun.
KITEW_EXACT = ("ld", "basis", "basis_kw0.1", "basis_kw10", "basis_kw100", "ld_basis",
               "ld_sdw10", "ld_sdw100", "ld_sdw1000")
KITEW_SAMPLED = ("ld", "basis", "basis_kw10", "basis_kw100", "ld_basis", "ld_sdw100", "ld_sdw1000")
LADDER_KITEW = [dict(policy="net", grad="exact", method=m, weights="onpolicy", lr=lr, sched="cos")
                for lr in (1e-3, 3e-3) for m in KITEW_EXACT]
LADDER_KITEW += [dict(policy="net", grad="sampled", method=m, weights="onpolicy", lr=1e-3, sched="cos")
                 for m in KITEW_SAMPLED]


def grid(name="sk"):
    if name == "kitew":
        return [dict(c, dag="free", D=10, T=T, J=J) for T in GRID_T for J in GRID_J for c in LADDER_KITEW] + \
               [dict(c, dag="free", kind="ising", D=9, T=T, J=J) for T in GRID_ISING_T
                for J in GRID_LDSD_SEEDS for c in LADDER_KITEW]
    if name == "ldsd3":
        return [dict(dict(dag="free", kind="ising"), **c, D=9, T=T, J=J) for T in GRID_ISING_T
                for J in GRID_LDSD_SEEDS for c in LADDER_LDSD]
    if name == "ising3":
        return [dict(dict(dag="free", kind="ising"), **c, D=9, T=T, J=J) for T in GRID_ISING_T
                for J in GRID_J for c in LADDER_ISING]
    return [dict(c, dag="free", D=D, T=T, J=J) for D in GRID_D for T in GRID_T for J in GRID_J
            for c in LADDER] + [dict(c, dag="free", D=D, T=T, J=J) for D in GRID_D for T in GRID_T
                                for J in GRID_J for c in LADDER2]


def task_name(t):
    w = t["weights"]
    tag = f"{t['method']}_{t['dag']}" + ("" if w == "onpolicy" else "_unif" if w == "uniform" else "_" + w)
    b = f"_B{t['B']}" if "B" in t else ""
    k = "ising3_" if t.get("kind") == "ising" else ""
    return f"{k}{t['policy']}_{t['grad']}_{tag}{b}_D{t['D']}_T{t['T']}_J{t['J']}_lr{t['lr']}_{t['sched']}"


def cmd_grid_run(a):
    tasks = grid(a.grid)
    ids = range(a.task * a.per_task, min(len(tasks), (a.task + 1) * a.per_task))
    os.makedirs(a.outdir, exist_ok=True)
    for i in ids:
        t = tasks[i]
        path = os.path.join(a.outdir, task_name(t) + ".json")
        if os.path.exists(path):
            continue
        r = train(t["D"], t["T"], t["J"], t["method"], t["dag"], t["lr"], t["sched"], a.iters, t["weights"],
                  policy=t["policy"], grad=t["grad"], B=t.get("B", 64), kind=t.get("kind", "sk"),
                  seed=t["J"] if t.get("kind") == "ising" else 0)
        with open(path + ".tmp", "w") as f:
            json.dump(r, f)
        os.replace(path + ".tmp", path)
        print(task_name(t), json.dumps(r["final"]), f"{r['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["perturb", "train", "grid-list", "grid-run"])
    p.add_argument("--D", type=int, default=6)
    p.add_argument("--T", type=float, default=0.4)
    p.add_argument("--J", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    p.add_argument("--method", default="basis")
    p.add_argument("--dag", default="free")
    p.add_argument("--weights", default="onpolicy")
    p.add_argument("--policy", default="table")
    p.add_argument("--grad", default="exact")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--sched", default="cos")
    p.add_argument("--iters", type=int, default=ITERS)
    p.add_argument("--task", type=int)
    p.add_argument("--per_task", type=int, default=1)
    p.add_argument("--outdir", default=None, help="default: outputs_kc_ladder (sk) / outputs_kc_<grid>")
    p.add_argument("--grid", default="sk", choices=["sk", "ising3", "ldsd3", "kitew"])
    p.add_argument("--kind", default="sk", choices=["sk", "ising"])
    a = p.parse_args()
    a.outdir = a.outdir or ("./outputs_kc_ladder" if a.grid == "sk" else f"./outputs_kc_{a.grid}")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    if a.command == "perturb":
        cmd_perturb(a)
    elif a.command == "train":
        r = train(a.D, a.T, a.J, a.method, a.dag, a.lr, a.sched, a.iters, a.weights,
                  policy=a.policy, grad=a.grad, kind=a.kind, seed=a.seed)
        for c in r["curve"]:
            print(json.dumps(c))
        print(f"{r['secs']:.1f}s")
    elif a.command == "grid-list":
        print(len(grid(a.grid)), "tasks")
    else:
        cmd_grid_run(a)
