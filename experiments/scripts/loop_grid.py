"""
loop_grid.py
============
Compute-matched comparison of the loop objectives (loop_gfn.py) against
TB / large-diamond baselines at small D, scored by EXACT TV to Boltzmann.
Decides which 2-3 configs get N=100 cluster time.

Subcommands:
  python loop_grid.py budget                  # once, writes <outdir>/budgets.json (seconds)
  python loop_grid.py dry_run                 # task counts + per-task time estimate
  python loop_grid.py run --task i            # one SLURM array task (skips if json exists)

Aggregate/plot: python make_plots.py loop_grid --indir outputs_loop_grid --outdir figures/loop_grid
"""
import argparse, copy, itertools, json, os, time
import numpy as np
import torch
import torch.nn as nn
from physics.spin_system import SpinEnv
from models.gflownet import TBGFlowNet
from utils.model_utils import make_batch_size_schedule
from experiments.scripts.loop_gfn import LoopGFlowNet, LOOP_PRESETS
#from experiments.scripts.loop_grid import CONFIGS as GRID_CONFIGS, _loop
from theory.loop_d8_exact_check import exact_model_dist, boltzmann

OUTDIR = "./outputs_loop_grid"
BUDGET_REF_ITERS = 4000
N_CKPT = 20  # TV every 5% of budget


def _loop(dag, preset, b=(16, 64), lo=None, hi=None):
    w_ld, w_kite, w_sd, plo, phi = LOOP_PRESETS[preset]
    return dict(method="LOOP", dag=dag, w_ld=w_ld, w_kite=w_kite, w_sd=w_sd,
                kite_lo=plo if lo is None else lo, kite_hi=phi if hi is None else hi,
                b0=b[0], b1=b[1])

TEMPS, J_SEEDS, TRAIN_SEEDS = [1.0, 0.4], [0, 1, 2, 3], [0, 1]
GRID_A = ["TB_tree", "ld_tree", "ld2x_tree", "kite_tree", "ld_kite_tree",
          "ld_free", "ld2x_free", "kite_free", "kite_sd_free", "basis_free"]
GRID_B = ["kite_early_tree", "kite_mid_tree", "kite_late_tree", "kite_tree"]
GRID_C = ["basis_free", "ld_free"]


CONFIGS = {
    # grid A, tree
    "TB_tree":         dict(method="TB", dag="tree", b0=16, b1=64),
    "ld_tree":         _loop("tree", "ld"),
    "ld2x_tree":       _loop("tree", "ld", b=(32, 128)),
    "kite_tree":       _loop("tree", "kite"),
    "ld_kite_tree":    _loop("tree", "ld_kite"),
    # grid A, free
    "ld_free":         _loop("free", "ld"),
    "ld2x_free":       _loop("free", "ld", b=(32, 128)),
    "kite_free":       _loop("free", "kite"),
    "kite_sd_free":    _loop("free", "kite_sd"),
    "basis_free":      _loop("free", "basis"),
    # grid B, kite depth bands (full band == kite_tree)
    "kite_early_tree": _loop("tree", "kite", lo=0.0, hi=0.33),
    "kite_mid_tree":   _loop("tree", "kite", lo=0.33, hi=0.67),
    "kite_late_tree":  _loop("tree", "kite", lo=0.67, hi=1.0),
}





def build_tasks():
    """Deduplicated list of (config, D, budget_mult); each is expanded over
    T x J_seed x train_seed. Order is fixed so array indices are stable."""
    cells = []
    for D in (8, 10):
        cells += [(c, D, 1) for c in GRID_A]
    cells += [(c, 10, 1) for c in GRID_B]
    for D in (6, 8, 10):
        for m in (1, 4):
            cells += [(c, D, m) for c in GRID_C]
    seen, uniq = set(), []
    for c in cells:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return [dict(config=c, D=D, mult=m, T=T, J_seed=j, train_seed=s)
            for (c, D, m) in uniq
            for T, j, s in itertools.product(TEMPS, J_SEEDS, TRAIN_SEEDS)]


def grids_of(t):
    g = []
    if t["mult"] == 1 and t["D"] in (8, 10) and t["config"] in GRID_A: g.append("A")
    if t["D"] == 10 and t["mult"] == 1 and t["config"] in GRID_B: g.append("B")
    if t["config"] in GRID_C: g.append("C")
    return g


def task_name(t):
    return f"{t['config']}_D{t['D']}_x{t['mult']}_T{t['T']}_J{t['J_seed']}_s{t['train_seed']}"


def make_model(cfg, env, T):
    if cfg["method"] == "TB":
        return TBGFlowNet(env, T)
    return LoopGFlowNet(env, T, dag=cfg["dag"], w_ld=cfg["w_ld"], w_kite=cfg["w_kite"],
                        w_sd=cfg["w_sd"], kite_lo=cfg["kite_lo"], kite_hi=cfg["kite_hi"],
                        n_sd=cfg.get("n_sd", 1))


def components(g, loss):
    c = dict(getattr(g, "last_components", {}) or {})
    c["total"] = float(loss.item())
    return c


def train_step(g, it, n_sched, B):
    loss = g.loss(it, n_sched, B)
    g.opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(g.policy.parameters(), 1.0)  # as in BatchedGFlowNetBase.train
    g.opt.step()
    return loss


def calibrate(g, cfg, n=30):
    """evals_grad per iteration at batch b0 and b1, on a throwaway copy
    (no optimizer step, so the real model and its counters are untouched).
    Uses the real loss() code path; consumes RNG, which is fine."""
    h = copy.deepcopy(g)
    per = []
    for B in (cfg["b0"], cfg["b1"]):
        e0 = h.evals_grad
        for _ in range(n):
            h.loss(1, 10, B)  # mid-anneal values; cost doesn't depend on them
        per.append((h.evals_grad - e0) / n)
    return per  # [cost/iter at b0, cost/iter at b1]


@torch.no_grad()
def exact_tv(g, p_true):
    cn, cg = g.evals_nograd, g.evals_grad
    p = exact_model_dist(g)
    g.evals_nograd, g.evals_grad = cn, cg
    return float(0.5 * np.abs(p - p_true).sum()), p


# ----------------------------------------------------------------
#  budget
# ----------------------------------------------------------------
class _ZeroPolicy(nn.Module):
    """Uniform logits. Only used by `budget`: kite_tree's evals_grad does not
    depend on the policy (base replay = B*D rows; partner replay starts at
    min(t_b), which is drawn independently of the policy), so this measures
    the exact same count through the real loss() code at a fraction of the cost."""
    def forward(self, spins, mask, step):
        return torch.zeros(spins.shape[0], 2, requires_grad=torch.is_grad_enabled())


def cmd_budget(a):
    os.makedirs(a.outdir, exist_ok=True)
    out = {}
    for D in (6, 8, 10):
        torch.manual_seed(0)
        env = SpinEnv("sk", D=D, p_af=0.5, J_seed=0)
        cfg = CONFIGS["kite_tree"]
        g = make_model(cfg, env, 1.0)
        g.policy = _ZeroPolicy()
        bs = make_batch_size_schedule(cfg["b0"], cfg["b1"])
        for it in range(1, BUDGET_REF_ITERS + 1):
            g.loss(it, BUDGET_REF_ITERS, bs(it, BUDGET_REF_ITERS))
        out[str(D)] = int(g.evals_grad)
        print(f"D={D}: BUDGET = {g.evals_grad} evals_grad "
              f"({g.evals_grad / BUDGET_REF_ITERS:.1f}/iter, nograd {g.evals_nograd})")
    with open(os.path.join(a.outdir, "budgets.json"), "w") as f:
        json.dump(out, f, indent=1)


def load_budgets(outdir):
    path = os.path.join(outdir, "budgets.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run `python loop_grid.py budget` first")
    return {int(k): v for k, v in json.load(open(path)).items()}


# ----------------------------------------------------------------
#  run one task
# ----------------------------------------------------------------
def cmd_run(a):
    tasks = build_tasks()
    t = tasks[a.task]
    out_path = os.path.join(a.outdir, task_name(t) + ".json")
    if os.path.exists(out_path):
        print(f"exists, skipping: {out_path}")
        return
    budget = load_budgets(a.outdir)[t["D"]] * t["mult"]
    cfg = CONFIGS[t["config"]]
    torch.manual_seed(t["train_seed"]); np.random.seed(t["train_seed"])
    env = SpinEnv("sk", D=t["D"], p_af=0.5, J_seed=t["J_seed"])
    g = make_model(cfg, env, t["T"])
    p_true = boltzmann(env, t["T"])

    c0, c1 = calibrate(g, cfg)
    # batch ramps linearly in iterations, cost/iter ~ linear between c0 and c1
    n_est = max(1, int(round(budget / (0.5 * (c0 + c1)))))
    bs = make_batch_size_schedule(cfg["b0"], cfg["b1"])

    tv0, _ = exact_tv(g, p_true)
    curve = [(0, 0, 0.0, tv0)]
    next_k, it, wall, comps_hist = 1, 0, 0.0, []
    while g.evals_grad < budget:
        it += 1
        t0 = time.perf_counter()
        loss = train_step(g, it, n_est, bs(it, n_est))
        wall += time.perf_counter() - t0
        comps_hist.append(components(g, loss))
        if g.evals_grad >= next_k * budget / N_CKPT or g.evals_grad >= budget:
            tv, p_model = exact_tv(g, p_true)
            curve.append((int(g.evals_grad), it, wall, tv))
            while next_k * budget / N_CKPT <= g.evals_grad:
                next_k += 1
            print(f"  {g.evals_grad / budget:5.0%}  it={it}  TV={tv:.4f}  "
                  f"loss={comps_hist[-1]['total']:.3g}", flush=True)

    tail = comps_hist[-max(1, len(comps_hist) // 100):]
    res = dict(
        **t, grids=grids_of(t), cfg=cfg, budget=budget, n_est=n_est,
        calib_grad_per_iter=[c0, c1], iters=it, anneal_frac=it / n_est,
        evals_grad=int(g.evals_grad), evals_nograd=int(g.evals_nograd),
        grad_per_iter=g.evals_grad / it, nograd_per_iter=g.evals_nograd / it,
        wall_train=wall, wall_per_iter=wall / it,
        tv_curve=curve, tv_curve_cols=["evals_grad", "iters", "wallclock", "tv"],
        final_tv=curve[-1][3],
        last_components=comps_hist[-1],
        tail_components={k: float(np.mean([c[k] for c in tail if k in c])) for k in tail[-1]},
        kl_true_model=float(np.sum(p_true * (np.log(p_true) - np.log(p_model + 1e-300)))),
        p_ground_true=float(p_true.max()), p_ground_model=float(p_model[p_true.argmax()]),
    )
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_path)
    print(f"wrote {out_path}  final TV={res['final_tv']:.4f}  iters={it} "
          f"(n_est {n_est}, anneal_frac {res['anneal_frac']:.3f})  wall={wall:.0f}s")


# ----------------------------------------------------------------
#  dry run: counts + per-task time estimate from a short local timing run
# ----------------------------------------------------------------
def cmd_dry_run(a):
    tasks = build_tasks()
    budgets = load_budgets(a.outdir)
    done = sum(os.path.exists(os.path.join(a.outdir, task_name(t) + ".json")) for t in tasks)
    print(f"{len(tasks)} tasks total, {done} already done, {len(tasks) - done} to run")
    for G in "ABC":
        print(f"  grid {G}: {sum(G in grids_of(t) for t in tasks)} tasks (overlaps counted per grid)")
    print(f"budgets: {budgets}\n")
    torch.set_num_threads(1)
    cells = sorted({(t["config"], t["D"], t["mult"]) for t in tasks}, key=lambda c: (c[1], c[0], c[2]))
    est = {}
    print(f"{'config':16s} {'D':>3s} {'x':>2s} {'grad/it@b0,b1':>16s} {'nograd/it':>10s} "
          f"{'ms/it':>7s} {'iters':>7s} {'TV s':>5s} {'est min':>8s}")
    for (c, D, m) in cells:
        cfg = CONFIGS[c]
        torch.manual_seed(0)
        env = SpinEnv("sk", D=D, p_af=0.5, J_seed=0)
        g = make_model(cfg, env, 0.4)
        c0, c1 = calibrate(g, cfg, n=10)
        Bm = (cfg["b0"] + cfg["b1"]) // 2
        n_time = 20
        n0 = g.evals_nograd
        t0 = time.perf_counter()
        for i in range(n_time):
            train_step(g, 1, 10, Bm)
        ms = (time.perf_counter() - t0) / n_time * 1e3
        ng = (g.evals_nograd - n0) / n_time
        t0 = time.perf_counter(); exact_tv(g, boltzmann(env, 0.4)); tv_s = time.perf_counter() - t0
        iters = budgets[D] * m / (0.5 * (c0 + c1))
        minutes = (iters * ms / 1e3 + (N_CKPT + 1) * tv_s) / 60
        est[(c, D, m)] = minutes
        print(f"{c:16s} {D:3d} {m:2d} {c0:7.0f},{c1:7.0f} {ng:10.0f} {ms:7.1f} {iters:7.0f} "
              f"{tv_s:5.2f} {minutes:8.1f}")
    per_task = [est[(t["config"], t["D"], t["mult"])] for t in tasks]
    print(f"\nper-task estimate (1 thread, head node): median {np.median(per_task):.1f} min, "
          f"max {max(per_task):.1f} min; total {sum(per_task) / 60:.1f} CPU-hours")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["budget", "dry_run", "run"])
    p.add_argument("--task", type=int)
    p.add_argument("--outdir", default=OUTDIR)
    a = p.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    {"budget": cmd_budget, "dry_run": cmd_dry_run, "run": cmd_run}[a.command](a)
