"""
loop_d8_exact_check.py
======================
Sanity check (HANDOFF step 3): train LOOP presets on a small SK instance
and compare the learned sampler to the exact Boltzmann distribution.

The TV distance is EXACT, not sampled: the trained policy's terminal
distribution is computed by pushing probability mass level by level
through every reachable partial state (2^(k) prefixes per level on the
tree, all 3^D partial assignments on the free DAG), so there is no
sampling noise in the metric.

One config per call (SLURM array task); results go to <outdir>/<name>.json.
  python loop_d8_exact_check.py --task 0 --outdir ...
  python loop_d8_exact_check.py --summarize --outdir ...
"""
import argparse, itertools, json, os, time
import numpy as np
import torch
from physics.spin_system import SpinEnv
from models.gflownet import TBGFlowNet
from utils.model_utils import make_batch_size_schedule
from experiments.scripts.loop_gfn import LoopGFlowNet, LOOP_PRESETS

# (label, method, dag, preset)
CONFIGS = [
    ("TB_tree",        "TB",   "tree", None),
    ("ld_tree",        "LOOP", "tree", "ld"),
    ("kite_tree",      "LOOP", "tree", "kite"),
    ("kite_last_tree", "LOOP", "tree", "kite_last"),   # expected negative control
    ("ld_free",        "LOOP", "free", "ld"),
    ("basis_free",     "LOOP", "free", "basis"),
]
TEMPS = [1.0, 0.4]
J_SEEDS = [0, 1]
GRID = list(itertools.product(range(len(CONFIGS)), TEMPS, J_SEEDS))


@torch.no_grad()
def exact_model_dist(g):
    """Exact P_model(x) over all 2^D terminals, indexed by binary code
    (spin +1 -> bit 1, site 0 most significant)."""
    D = g.env.D
    free = isinstance(g, LoopGFlowNet) and g.dag == "free"
    level = {(0,) * D: 1.0}
    for t in range(D):
        keys = list(level)
        S = torch.tensor(keys, dtype=torch.get_default_dtype())
        M = (S != 0).to(S.dtype)
        if isinstance(g, LoopGFlowNet):
            P = g._logp(S, M, t).exp()
        else:
            P = torch.softmax(g.policy(S, M, t), dim=-1)
        P = P.tolist()  # python floats: far faster than per-element .item()
        nxt = {}
        for r, k in enumerate(keys):
            m = level[k]
            for a, p in enumerate(P[r]):
                if p == 0.0:
                    continue
                site, v = (a // 2, a % 2) if free else (t, a)
                k2 = list(k); k2[site] = 1 if v == 1 else -1
                k2 = tuple(k2)
                nxt[k2] = nxt.get(k2, 0.0) + m * p
        level = nxt
    out = np.zeros(2 ** D)
    for k, m in level.items():
        out[int("".join("1" if s > 0 else "0" for s in k), 2)] = m
    return out


def boltzmann(env, T):
    X = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=env.D)))
    return torch.softmax(-env.energy_batch(X) / T, 0).numpy()


def run(task, D, n_iter, outdir):
    ci, T, js = GRID[task]
    label, method, dag, preset = CONFIGS[ci]
    torch.manual_seed(1000 + task); np.random.seed(1000 + task)
    env = SpinEnv("sk", D=D, p_af=0.5, J_seed=js)
    if method == "TB":
        g = TBGFlowNet(env, T)
    else:
        w_ld, w_kite, w_sd, lo, hi = LOOP_PRESETS[preset]
        g = LoopGFlowNet(env, T, dag=dag, w_ld=w_ld, w_kite=w_kite, w_sd=w_sd,
                         kite_lo=lo, kite_hi=hi)
    t0 = time.time()
    loss_log = []
    g.train(n_iter, batch_size_fn=make_batch_size_schedule(16, 64), loss_log=loss_log,
            log_every=250)
    p_model = exact_model_dist(g)
    p_true = boltzmann(env, T)
    res = dict(label=label, method=method, dag=dag, preset=preset, T=T, J_seed=js, D=D,
               n_iter=n_iter, tv=float(0.5 * np.abs(p_model - p_true).sum()),
               mass=float(p_model.sum()),
               kl_true_model=float(np.sum(p_true * (np.log(p_true) - np.log(p_model + 1e-300)))),
               p_ground_true=float(p_true.max()), p_ground_model=float(p_model[p_true.argmax()]),
               secs=time.time() - t0, loss_log=loss_log)
    name = f"{label}_T{T}_J{js}.json"
    with open(os.path.join(outdir, name), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "loss_log"}))


def summarize(outdir):
    rows = [json.load(open(os.path.join(outdir, f))) for f in sorted(os.listdir(outdir))
            if f.endswith(".json")]
    print(f"{len(rows)}/{len(GRID)} results\n")
    print(f"{'config':16s}" + "".join(f"  T={T}: TV(J0, J1)      " for T in TEMPS))
    for label, *_ in CONFIGS:
        line = f"{label:16s}"
        for T in TEMPS:
            tvs = [r["tv"] for js in J_SEEDS for r in rows
                   if r["label"] == label and r["T"] == T and r["J_seed"] == js]
            line += "  " + (", ".join(f"{v:.3f}" for v in tvs) if tvs else "-").ljust(22)
        print(line)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=int)
    p.add_argument("--D", type=int, default=8)
    p.add_argument("--n_iter", type=int, default=4000)
    p.add_argument("--outdir", default="./outputs_loop_d8")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.summarize:
        summarize(a.outdir)
    else:
        run(a.task, a.D, a.n_iter, a.outdir)
