"""
Correctness tests for loop_gfn.py.

Builds the EXACT optimal forward policy by enumeration on a small SK
instance, for both DAGs, and checks that every loop loss is ~0 there and
clearly nonzero after perturbing the policy.

  tree: F(s) = sum_{x consistent with s} R(x),  P_F(v|s) = F(s+v)/F(s)
  free (uniform P_B over assigned sites):
        F(s) = sum_{x consistent with s} R(x) / C(D, k)     (k = #assigned)
        P_F(s -> s+(i,v)) = F(s+(i,v)) / (F(s) * (k+1))

Run: python test_loop_gfn.py
"""
import itertools, math
import numpy as np
import torch
import torch.nn as nn

#from archive.spin_glass_lib import SpinEnv
from physics.spin_system import SpinEnv
from experiments.scripts.loop_gfn import LoopGFlowNet

torch.set_default_dtype(torch.float64)
D, T = 5, 0.7

def all_configs(D):
    return torch.tensor(list(itertools.product([-1.0, 1.0], repeat=D)))

class ExactPolicy(nn.Module):
    def __init__(self, env, T, dag, noise=0.0):
        super().__init__()
        self.dag, self.D = dag, env.D
        X = all_configs(env.D)
        self.X = X
        self.logR = (-env.energy_batch(X) / T)
        self.noise = noise
        self.dummy = nn.Parameter(torch.zeros(1))
        g = torch.Generator().manual_seed(0)
        self.noise_table = torch.randn(10_000, generator=g)

    def logF(self, spins, mask):
        # consistent[b, x] = all assigned sites of b agree with config x
        agree = ((spins.unsqueeze(1) == self.X.unsqueeze(0)) | (mask.unsqueeze(1) < 0.5)).all(-1)
        lf = torch.logsumexp(self.logR.unsqueeze(0).masked_fill(~agree, float("-inf")), dim=1)
        if self.dag == "free":
            k = mask.sum(-1)
            lf = lf - torch.tensor([math.log(math.comb(self.D, int(kk))) for kk in k])
        return lf

    def _noise(self, spins, mask, n_out):
        h = ((spins + 2) * mask * torch.arange(1, self.D + 1)).sum(-1).long()
        idx = (h.unsqueeze(1) * 7 + torch.arange(n_out)) % 10_000
        return self.noise * self.noise_table[idx]

    def forward(self, spins, mask, step):
        B = spins.shape[0]
        lf_s = self.logF(spins, mask)
        if self.dag == "tree":
            t = int(step)
            out = []
            for v in (-1.0, 1.0):
                sp, mk = spins.clone(), mask.clone()
                sp[:, t] = v; mk[:, t] = 1.0
                out.append(self.logF(sp, mk) - lf_s)
            logits = torch.stack(out, 1)
        else:
            k = mask.sum(-1)
            out = []
            for i in range(self.D):
                for v in (-1.0, 1.0):
                    sp, mk = spins.clone(), mask.clone()
                    sp[:, i] = v; mk[:, i] = 1.0
                    val = self.logF(sp, mk) - lf_s - torch.log(k + 1)
                    val = torch.where(mask[:, i] > 0.5, torch.full_like(val, -50.0), val)
                    out.append(val)
            logits = torch.stack(out, 1)
        return logits + self._noise(spins, mask, logits.shape[1])


def run_case(dag, preset_kwargs, noise):
    torch.manual_seed(1)
    env = SpinEnv("sk", D=D, p_af=0.5, J_seed=3)
    g = LoopGFlowNet(env, T, dag=dag, **preset_kwargs)
    g.policy = ExactPolicy(env, T, dag, noise=noise)
    # it = n_iter -> beta = 1 (true target), entropy coef = 0
    with torch.no_grad():
        L = g.loss(1000, 1000, 64)
    return float(L), g.last_components


def check_marginal(dag):
    """
    Sampling with the exact policy reproduces Boltzmann (sanity check on
    _rollout / _apply for both DAGs).
    """
    torch.manual_seed(2)
    env = SpinEnv("sk", D=D, p_af=0.5, J_seed=3)
    g = LoopGFlowNet(env, T, dag=dag)
    g.policy = ExactPolicy(env, T, dag)
    n = 20000
    acts = g._rollout(n, eps=0.0)
    _, _, spins = g._replay(acts)
    X = all_configs(D)
    p_true = torch.softmax(-env.energy_batch(X) / T, 0)
    keys = ((spins > 0).long() * (2 ** torch.arange(D - 1, -1, -1))).sum(1)
    emp = torch.bincount(keys, minlength=2 ** D).double() / n
    return 0.5 * (emp - p_true).abs().sum().item()


def check_grads_finite(dag, kw, steps=5):
    """A few real training steps (beta < 1, entropy on): every grad finite.
    Guards the -inf illegal-action logits on the free DAG."""
    torch.manual_seed(3)
    env = SpinEnv("sk", D=D, p_af=0.5, J_seed=3)
    g = LoopGFlowNet(env, T, dag=dag, **kw)
    for it in range(steps):
        L = g.loss(it, 1000, 32)
        g.opt.zero_grad(); L.backward()
        if not all(torch.isfinite(p.grad).all() for p in g.policy.parameters() if p.grad is not None):
            return False
        g.opt.step()
    return True


if __name__ == "__main__":
    cases = [
        ("tree", dict(w_ld=1, w_kite=0, w_sd=0)),
        ("tree", dict(w_ld=0, w_kite=1, w_sd=0, kite_lo=0, kite_hi=1, n_kite=2)),
        ("tree", dict(w_ld=0, w_kite=1, w_sd=0, kite_lo=1, kite_hi=1)),
        ("free", dict(w_ld=1, w_kite=0, w_sd=0)),
        ("free", dict(w_ld=0, w_kite=1, w_sd=0, kite_lo=0, kite_hi=1, n_kite=2)),
        ("free", dict(w_ld=0, w_kite=1, w_sd=0, kite_lo=0, kite_hi=1, kite_diverge="any")),
        ("free", dict(w_ld=0, w_kite=1, w_sd=0, kite_lo=1, kite_hi=1)),
        ("free", dict(w_ld=0, w_kite=0, w_sd=1, n_sd=3)),
    ]
    ok = True
    for dag, kw in cases:
        L0, c0 = run_case(dag, kw, noise=0.0)
        L1, c1 = run_case(dag, kw, noise=0.5)
        passed = L0 < 1e-8 and L1 > 1e-3
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  dag={dag:4s} {kw}  exact={L0:.2e}  perturbed={L1:.3e}")
    for dag in ("tree", "free"):
        tv = check_marginal(dag)
        passed = tv < 0.03
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  exact-policy sampling TV to Boltzmann, dag={dag}: {tv:.4f}")
    for dag, kw in [("tree", dict(w_ld=1, w_kite=1)), ("free", dict(w_ld=1, w_kite=1, w_sd=1))]:
        passed = check_grads_finite(dag, kw)
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  finite grads in training steps, dag={dag}")
    print("\nALL PASS" if ok else "\nSOME FAILED")
