"""
diagnose_loops.py (rename to calulate_residuals.py?)
=================
Why do the local loop objectives (basis = one-step kites + small diamonds)
fail on the 10x10 free DAG? Loads trained policies from outputs_loop_big/
and measures, at beta = 1, every residual the theory says should vanish:

  kite   r_k = [lp(a|s) - lp(a^1|s)] - [logR(x) - logR(x^flip)]   (last step)
  sd(t)  r_sd = [lp(a|s) + lp(b|s+a)] - [lp(b|s) + lp(a|s+b)]     (every depth t)
  order  std over K random orders of log P_F(tau | x)              (path dependence of q)
  V      std over trajectories of V = log P_F(tau) - logR(x)       (the global KC residual)

each on two trajectory sets:
  q  on-policy rollouts (what the loss is actually trained on)
  p  Wolff samples from the target, random site orders (where it should also hold)

Usage: python diagnose_loops.py [--n 500] [--T 1.5 2.27 3.2] [--configs ...]
"""
import argparse, json, os
import numpy as np
import torch
from physics.spin_system import SpinEnv
from models.mcmc import run_mcmc
from experiments.scripts.loop_big import CONFIGS, task_name
from experiments.scripts.loop_grid import make_model

L = 10

def residuals(g, acts):
    """Per-trajectory residuals for actions acts [n, D] at beta = 1."""
    n, D = acts.shape
    lp, _, x = g._replay(acts)
    logR = g._logR(x, 1.0)
    V = lp.sum(1) - logR

    # one-step kite at the last step: flip the last-assigned spin
    s, m = g._prefix_state(acts, torch.full((n,), D - 1))
    lpl = g._logp(s, m, D - 1)
    a = acts[:, -1]
    xf = x.clone()
    site = a // 2 if g.dag == "free" else torch.full_like(a, D - 1)
    xf[torch.arange(n), site] *= -1
    r_k = (lpl.gather(1, a[:, None]) - lpl.gather(1, (a ^ 1)[:, None])).squeeze(1) \
        - (logR - g._logR(xf, 1.0))

    out = dict(V=V, r_k=r_k)
    if g.dag == "free":
        # small diamond at every depth t = 0..D-2
        rows = torch.arange(n).repeat_interleave(D - 1)
        t = torch.arange(D - 1).repeat(n)
        A = acts[rows]
        a, b = A[torch.arange(len(t)), t], A[torch.arange(len(t)), t + 1]
        orig = lp[rows, t] + lp[rows, t + 1]
        s, m = g._prefix_state(A, t)
        lps = g._logp(s, m, t)
        s2, m2 = s.clone(), m.clone()
        i = torch.arange(len(t))
        s2[i, b // 2] = torch.where(b % 2 == 0, -1.0, 1.0)
        m2[i, b // 2] = 1.0
        lpb = g._logp(s2, m2, t + 1)
        swap = lps.gather(1, b[:, None]).squeeze(1) + lpb.gather(1, a[:, None]).squeeze(1)
        out["r_sd"] = (orig - swap).view(n, D - 1)
    return out


def random_order_acts(X, D):
    perm = torch.argsort(torch.rand(len(X), D), dim=1)
    return 2 * perm + (X.gather(1, perm) > 0).long()


def order_spread(g, X, K=16):
    """std over K random orders of log P_F(tau|x), per x (free DAG only)."""
    lps = torch.stack([g._replay(random_order_acts(X, g.env.D))[0].sum(1) for _ in range(K)], 1)
    return lps.std(1)


def summarize(res):
    rms = lambda v: float(v.pow(2).mean().sqrt())
    o = dict(V_std=float(res["V"].std()), kite_rms=rms(res["r_k"]), kite_mean=float(res["r_k"].mean()))
    if "r_sd" in res:
        r = res["r_sd"]
        D1 = r.shape[1]
        bands = [(0, D1 // 4), (D1 // 4, D1 // 2), (D1 // 2, 3 * D1 // 4), (3 * D1 // 4, D1)]
        o["sd_rms_by_depth"] = [rms(r[:, a:b]) for a, b in bands]
        o["sd_mean_by_depth"] = [float(r[:, a:b].mean()) for a, b in bands]
        # one trajectory's summed diamond residual (signed); a random-sign
        # walk of D-1 terms would give sqrt(D-1) * rms
        o["sd_sum_std"] = float(r.sum(1).std())
        o["sd_rms"] = rms(r)
    return o

@torch.no_grad()
def main(a):
    torch.set_num_threads(4)
    torch.manual_seed(0); np.random.seed(0)
    env = SpinEnv("ising", N=L)
    D = env.D
    report = {}
    for T in a.T:
        Xp = torch.as_tensor(np.asarray(run_mcmc(env, T, a.n, warmup=2000, thin=10, method="wolff")),
                             dtype=torch.get_default_dtype())
        for cfg in a.configs:
            for mult in (1, 4):
                name = task_name(dict(kind="gfn", config=cfg, T=T, train_seed=a.seed, mult=mult,
                                      match=a.match))
                path = os.path.join(a.outputs, name + ".policy.pt")
                if not os.path.exists(path):
                    continue
                g = make_model(CONFIGS[cfg], env, T)
                g.policy.load_state_dict(torch.load(path, weights_only=True))
                g.policy.eval()
                acts_q = g._rollout(a.n, eps=0.0)
                Xq = g._replay(acts_q)[2]
                acts_p = random_order_acts(Xp, D) if g.dag == "free" else (Xp > 0).long()
                row = dict(q=summarize(residuals(g, acts_q)), p=summarize(residuals(g, acts_p)),
                           KL=json.load(open(path.replace(".policy.pt", ".json")))["final"]["kl_q_p"])
                if g.dag == "free":
                    row["q"]["order_std"] = float(order_spread(g, Xq[:200]).mean())
                    row["p"]["order_std"] = float(order_spread(g, Xp[:200]).mean())
                report[name] = row
                print(name, json.dumps(row), flush=True)
    os.makedirs("figures/loop_big", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--T", type=float, nargs="+", default=[1.5, 2.27, 3.2])
    p.add_argument("--configs", nargs="+", default=["basis_free", "ld_free", "kite_free", "ld_tree"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outputs", default="outputs_loop_big")
    p.add_argument("--match", default=None, help="'iters' for the sdk suite task names")
    p.add_argument("--out", default="figures/loop_big/diagnose_loops.json")
    main(p.parse_args())
