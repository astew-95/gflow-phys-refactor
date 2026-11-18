"""
loop_big.py
===========
Big compute-matched runs of the loop objectives on the 10x10 periodic
Ising ferromagnet (D = 100), against two references:
  - EXACT (Kaufman, ising_exact.py): log Z, <E>, C at every T;
  - a long Wolff MCMC run per T (its own task): <|m|>, chi, U4.

Configs (loop_gfn presets; see HANDOFF.md):
  ld_tree     large diamonds = KC, fixed order          (anchor to all earlier runs)
  ld_free     large diamonds, random-order DAG
  kite_free   kites, full depth band
  ld_sd_free  large + small diamonds                     ("do small diamonds help KC?")
  basis_free  one-step kites + small diamonds            (DB-like end; expected to lag)

Subcommands:
  python loop_big.py pilot  --config C --T 2.27 --iters 40   # timing (submit via SLURM)
  python loop_big.py list                                    # task table
  python loop_big.py run --task i                            # one array task
  python loop_big.py summarize
"""
import argparse, json, math, os, time
import numpy as np
import torch

from physics.spin_system import SpinEnv
from physics.observables import magnetization_susceptibility
from models.mcmc import run_mcmc
from utils.model_utils import make_batch_size_schedule

#from archive.spin_glass_lib import (SpinEnv, make_batch_size_schedule, run_mcmc, magnetization_susceptibility)

from experiments.scripts.loop_grid import CONFIGS as GRID_CONFIGS, _loop, make_model, calibrate, train_step, components
import theory.ising_exact as ising_exact

OUTDIR = "./outputs_loop_big"

L = 10
CONFIGS = {
    "ld_tree": GRID_CONFIGS["ld_tree"],
    "ld_free": GRID_CONFIGS["ld_free"],
    "kite_free": GRID_CONFIGS["kite_free"],
    "ld_sd_free": _loop("free", "ld_sd"),
    "basis_free": GRID_CONFIGS["basis_free"],
}
T_GRID = [1.5, 1.8, 2.0, 2.1, 2.2, 2.27, 2.35, 2.45, 2.6, 2.8, 3.0, 3.2]
TRAIN_SEEDS = [0, 1, 2]
N_CKPT = 20          # light eval + checkpoint every 5% of budget
N_EVAL = 4000        # final samples
N_EVAL_LIGHT = 500   # samples per intermediate eval
K_ORDERS = 16        # random orders per sample for log q on the free DAG
MCMC = dict(n=20000, warmup=2000, thin=10)  # Wolff units = cluster flips


X4_CONFIGS = ["basis_free", "ld_free"]  # propagation check at D=100 (mirrors loop_grid C)
SDK = (2, 8, 32)                         # small diamonds per large diamond (suite "sdk")
for _k in SDK:
    CONFIGS[f"ld_sd{_k}_free"] = dict(_loop("free", "ld_sd"), n_sd=_k // 2)
# Suite "sdw" (-> outputs_loop_sdw/): the sdk diamond loss is a MEAN over diamonds, so K changes
# only the noise of the sd term, never its weight (sd ~1e-3 vs ld ~3 at w_sd=1). Here K=32 with
# w_sd raised until the sd term is comparable to ld. Iteration-matched like sdk.
SDW = (10, 100, 1000)
SDW_T = [1.5, 2.0, 2.27, 2.6, 3.2]
for _w in SDW:
    CONFIGS[f"ld_sd32w{_w}_free"] = dict(_loop("free", "ld_sd"), n_sd=16, w_sd=float(_w))
SUITE = "main"
SUITE_DIRS = {"main": OUTDIR, "sdk": "./outputs_loop_sdk", "sdw": "./outputs_loop_sdw"}


def build_tasks():
    """Fixed order (array indices are stable): x1 GFN tasks, Wolff tasks, x4 tasks."""
    if SUITE == "sdk":
        return [dict(kind="gfn", config=f"ld_sd{k}_free", T=T, train_seed=s, mult=1, match="iters")
                for k in SDK for T in T_GRID for s in TRAIN_SEEDS]
    if SUITE == "sdw":
        return [dict(kind="gfn", config=f"ld_sd32w{w}_free", T=T, train_seed=s, mult=1, match="iters")
                for w in SDW for T in SDW_T for s in TRAIN_SEEDS]
    main = ["ld_tree", "ld_free", "kite_free", "ld_sd_free", "basis_free"]
    tasks = [dict(kind="gfn", config=c, T=T, train_seed=s, mult=1)
             for c in main for T in T_GRID for s in TRAIN_SEEDS]
    tasks += [dict(kind="mcmc", config="wolff", T=T, train_seed=0, mult=1) for T in T_GRID]
    tasks += [dict(kind="gfn", config=c, T=T, train_seed=s, mult=4)
              for c in X4_CONFIGS for T in T_GRID for s in TRAIN_SEEDS]
    return tasks


def task_name(t):
    if t["kind"] == "mcmc":
        return f"{t['config']}_L{L}_T{t['T']}"
    m = "it" if t.get("match") == "iters" else "x"
    return f"{t['config']}_L{L}_{m}{t['mult']}_T{t['T']}_s{t['train_seed']}"


def budget_for(ref_iters):
    bs = make_batch_size_schedule(16, 64)
    return L * L * sum(bs(it, ref_iters) for it in range(1, ref_iters + 1))


# ----------------------------------------------------------------
#  evaluation
# ----------------------------------------------------------------
def observables(env, X, T):
    """X: [n, D] numpy +-1. Physics observables, D-normalized where usual."""
    Xt = torch.as_tensor(X, dtype=torch.get_default_dtype())
    E = env.energy_batch(Xt).numpy().astype(np.float64)
    m = X.mean(1)
    chi, absm = magnetization_susceptibility(env, X, T)
    m2, m4 = float((m ** 2).mean()), float((m ** 4).mean())
    return dict(E_per_spin=float(E.mean() / env.D), C_per_spin=float(E.var() / (env.D * T * T)),
                abs_m=absm, chi=float(chi), U4=1.0 - m4 / (3 * m2 * m2 + 1e-300),
                frac_m_pos=float((m > 0).mean()), frac_m_zero=float((m == 0).mean())), E


@torch.no_grad()
def log_q(g, X, K=K_ORDERS, chunk=4096):
    """log q(x) per sample: exact on the tree, K-order IS estimate on the free DAG."""
    D = g.env.D
    Xt = torch.as_tensor(X, dtype=torch.get_default_dtype())
    if g.dag == "tree":
        acts = (Xt > 0).long()
        out = []
        for i in range(0, len(acts), chunk):
            lp, _, _ = g._replay(acts[i:i + chunk])
            out.append(lp.sum(1))
        return torch.cat(out).numpy()
    n = len(Xt)
    lpf = torch.empty(n, K)
    for k in range(K):
        perm = torch.argsort(torch.rand(n, D), dim=1)          # random site order per sample
        vals = (Xt.gather(1, perm) > 0).long()
        acts = 2 * perm + vals
        for i in range(0, n, chunk):
            lp, _, _ = g._replay(acts[i:i + chunk])
            lpf[i:i + chunk, k] = lp.sum(1)
    log_pb = -math.lgamma(D + 1)
    return (torch.logsumexp(lpf, 1) - math.log(K) - log_pb).numpy()


@torch.no_grad()
def evaluate(g, env, T, n, with_logq):
    cn, cg = g.evals_nograd, g.evals_grad
    X = np.stack(g.sample(n))
    obs, E = observables(env, X, T)
    if with_logq:
        lq = log_q(g, X)
        logR = -E / T
        lz = ising_exact.log_Z(L, T)
        w = logR - lq
        lz_is = float(np.logaddexp.reduce(w) - math.log(len(w)))
        ess = float(np.exp(2 * np.logaddexp.reduce(w) - np.logaddexp.reduce(2 * w)) / len(w))
        obs.update(kl_q_p=float(np.mean(lq - logR) + lz), logZ_is=lz_is,
                   logZ_is_err=lz_is - lz, ess_frac=ess,
                   logq_estimator="exact" if g.dag == "tree" else f"IS_{K_ORDERS}_orders")
    g.evals_nograd, g.evals_grad = cn, cg
    return obs


def exact_ref(T):
    return dict(logZ=ising_exact.log_Z(L, T),
                E_per_spin=ising_exact.mean_energy(L, T) / (L * L),
                C_per_spin=ising_exact.specific_heat(L, T) / (L * L))


# ----------------------------------------------------------------
#  tasks
# ----------------------------------------------------------------
def run_mcmc_task(t, out_path):
    env = SpinEnv("ising", N=L)
    np.random.seed(1234); torch.manual_seed(1234)
    t0 = time.time()
    X, diag = run_mcmc(env, t["T"], MCMC["n"], warmup=MCMC["warmup"], thin=MCMC["thin"],
                       method="wolff", return_diagnostics=True)
    X = np.asarray(X, dtype=np.float64)
    obs, _ = observables(env, X, t["T"])
    return dict(**t, L=L, mcmc=MCMC, diagnostics={k: (float(v) if not isinstance(v, str) else v)
                                                   for k, v in diag.items()},
                final=obs, exact=exact_ref(t["T"]), secs=time.time() - t0)


def run_gfn_task(t, out_path, ref_iters):
    cfg = CONFIGS[t["config"]]
    budget = budget_for(ref_iters) * t["mult"]
    by_iters = t.get("match") == "iters"
    torch.manual_seed(t["train_seed"]); np.random.seed(t["train_seed"])
    env = SpinEnv("ising", N=L)
    g = make_model(cfg, env, t["T"])
    ckpt_path = out_path.replace(".json", ".ckpt.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        g.policy.load_state_dict(ck["policy"]); g.opt.load_state_dict(ck["opt"])
        g.evals_grad, g.evals_nograd = ck["evals_grad"], ck["evals_nograd"]
        torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["np_rng"])
        st = ck["state"]
    else:
        c0, c1 = calibrate(g, cfg)
        n_est = ref_iters * t["mult"] if by_iters else max(1, int(round(budget / (0.5 * (c0 + c1)))))
        st = dict(n_est=n_est, calib=[c0, c1], it=0, wall=0.0, next_k=1, curve=[], comps_tail=[])
    # progress in [0, 1]: iterations (iteration-matched) or evals_grad (compute-matched)
    prog = (lambda: st["it"] / st["n_est"]) if by_iters else (lambda: g.evals_grad / budget)
    if os.path.exists(ckpt_path):
        print(f"resumed from {ckpt_path} at {prog():.0%}")
    bs = make_batch_size_schedule(cfg["b0"], cfg["b1"])
    while prog() < 1:
        st["it"] += 1
        t0 = time.perf_counter()
        loss = train_step(g, st["it"], st["n_est"], bs(st["it"], st["n_est"]))
        st["wall"] += time.perf_counter() - t0
        st["comps_tail"] = (st["comps_tail"] + [components(g, loss)])[-200:]
        if prog() >= st["next_k"] / N_CKPT or prog() >= 1:
            ob = evaluate(g, env, t["T"], N_EVAL_LIGHT, with_logq=False)
            st["curve"].append(dict(evals_grad=int(g.evals_grad), iters=st["it"], wall=st["wall"],
                                    loss=st["comps_tail"][-1]["total"], **ob))
            while st["next_k"] / N_CKPT <= prog():
                st["next_k"] += 1
            print(f"  {prog():5.0%} it={st['it']} E/D={ob['E_per_spin']:.4f} "
                  f"|m|={ob['abs_m']:.3f} m>0={ob['frac_m_pos']:.2f} loss={st['curve'][-1]['loss']:.3g} "
                  f"wall={st['wall']:.0f}s", flush=True)
            if prog() < 1:
                tmp = ckpt_path + ".tmp"
                torch.save(dict(policy=g.policy.state_dict(), opt=g.opt.state_dict(),
                                evals_grad=g.evals_grad, evals_nograd=g.evals_nograd,
                                torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(),
                                state=st), tmp)
                os.replace(tmp, ckpt_path)
    t0 = time.perf_counter()
    final = evaluate(g, env, t["T"], N_EVAL, with_logq=True)
    tail = st["comps_tail"][-max(1, len(st["comps_tail"]) // 4):]
    res = dict(**t, L=L, cfg=cfg, ref_iters=ref_iters, budget=budget, n_est=st["n_est"],
               calib_grad_per_iter=st["calib"], iters=st["it"], anneal_frac=st["it"] / st["n_est"],
               evals_grad=int(g.evals_grad), evals_nograd=int(g.evals_nograd),
               wall_train=st["wall"], wall_eval=time.perf_counter() - t0,
               curve=st["curve"], final=final, exact=exact_ref(t["T"]),
               last_components=st["comps_tail"][-1],
               tail_components={k: float(np.mean([c[k] for c in tail if k in c])) for k in tail[-1]})
    torch.save(g.policy.state_dict(), out_path.replace(".json", ".policy.pt"))
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return res


def cmd_run(a):
    t = build_tasks()[a.task]
    os.makedirs(a.outdir, exist_ok=True)
    out_path = os.path.join(a.outdir, task_name(t) + ".json")
    if os.path.exists(out_path):
        print(f"exists, skipping: {out_path}")
        return
    print(f"task {a.task}: {t}")
    res = run_mcmc_task(t, out_path) if t["kind"] == "mcmc" else run_gfn_task(t, out_path, a.ref_iters)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_path)
    f = res["final"]
    print(f"wrote {out_path}: E/D={f['E_per_spin']:.4f} (exact {res['exact']['E_per_spin']:.4f}) "
          f"|m|={f['abs_m']:.3f} chi={f['chi']:.3f} m>0={f['frac_m_pos']:.3f}"
          + (f" KL={f['kl_q_p']:.3f} ESS={f['ess_frac']:.3f}" if "kl_q_p" in f else ""))


def cmd_list(a):
    tasks = build_tasks()
    for i, t in enumerate(tasks):
        done = os.path.exists(os.path.join(a.outdir, task_name(t) + ".json"))
        print(f"{i:4d}  {'done' if done else '    '}  {task_name(t)}")
    print(f"\n{len(tasks)} tasks; budget(ref_iters={a.ref_iters}) = {budget_for(a.ref_iters):,} evals_grad")


def cmd_pilot(a):
    """Time a few training iterations + one full evaluation for one config at
    D=100, and extrapolate to the budget. Heavy enough to belong on SLURM."""
    cfg = CONFIGS[a.config]
    torch.manual_seed(0)
    env = SpinEnv("ising", N=L)
    g = make_model(cfg, env, a.T)
    c0, c1 = calibrate(g, cfg, n=5)
    budget = budget_for(a.ref_iters)
    n_est = budget / (0.5 * (c0 + c1))
    Bm = (cfg["b0"] + cfg["b1"]) // 2
    t0 = time.perf_counter()
    for _ in range(a.iters):
        train_step(g, 1, 10, Bm)
    s_it = (time.perf_counter() - t0) / a.iters
    t0 = time.perf_counter(); evaluate(g, env, a.T, N_EVAL_LIGHT, with_logq=False); s_light = time.perf_counter() - t0
    t0 = time.perf_counter(); fin = evaluate(g, env, a.T, N_EVAL, with_logq=True); s_final = time.perf_counter() - t0
    total = n_est * s_it + N_CKPT * s_light + s_final
    print(json.dumps(dict(config=a.config, threads=torch.get_num_threads(), grad_per_iter=[c0, c1],
                          n_est=round(n_est), s_per_iter=s_it, s_light_eval=s_light,
                          s_final_eval=s_final, est_task_hours=total / 3600,
                          untrained_final=fin)))


def cmd_summarize(a):
    rows = [json.load(open(os.path.join(a.outdir, f))) for f in sorted(os.listdir(a.outdir))
            if f.endswith(".json")]
    n_suite = len(rows)
    if SUITE in ("sdk", "sdw"):   # ld_free (= K 0) and the Wolff references come from the main suite
        rows += [json.load(open(os.path.join(OUTDIR, f))) for f in sorted(os.listdir(OUTDIR))
                 if f.endswith(".json") and (f.startswith("ld_free_L10_x1_") or f.startswith("wolff_"))]
    ref = {r["T"]: r["final"] for r in rows if r["kind"] == "mcmc"}
    keys = ["E_per_spin", "abs_m", "chi", "U4", "frac_m_pos", "kl_q_p", "ess_frac"]
    print(f"{n_suite}/{len(build_tasks())} results")
    for T in T_GRID:
        ex = exact_ref(T)
        w = ref.get(T, {})
        print(f"\nT={T}  exact E/D={ex['E_per_spin']:.4f}  Wolff: |m|={w.get('abs_m', float('nan')):.3f} "
              f"chi={w.get('chi', float('nan')):.3f} U4={w.get('U4', float('nan')):.3f}")
        cs = {"sdk": ["ld_free"] + [f"ld_sd{k}_free" for k in SDK],
              "sdw": ["ld_free"] + [f"ld_sd32w{w}_free" for w in SDW]}.get(SUITE, CONFIGS)
        for c, m in [(c, m) for c in cs for m in (1, 4)]:
            cell = [r["final"] for r in rows if r["kind"] == "gfn" and r["config"] == c
                    and r["T"] == T and r.get("mult", 1) == m]
            if not cell:
                continue
            line = f"  {c:11s} x{m} n={len(cell)}"
            for k in keys:
                v = np.array([x.get(k, np.nan) for x in cell])
                line += f"  {k}={v.mean():.3f}±{v.std():.3f}"
            print(line)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["pilot", "list", "run", "summarize"])
    p.add_argument("--task", type=int)
    p.add_argument("--config", default="ld_tree", choices=sorted(CONFIGS))
    p.add_argument("--T", type=float, default=2.27)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--ref_iters", type=int, default=8000,
                   help="budget = ld_tree's evals_grad over this many iterations (B 16->64)")
    p.add_argument("--outdir", default=None, help="default: outputs_loop_big (main) / outputs_loop_sdk")
    p.add_argument("--suite", default="main", choices=["main", "sdk", "sdw"])
    p.add_argument("--L", type=int, default=L, help="lattice side (default 10; smaller only for testing)")
    a = p.parse_args()
    L = a.L
    SUITE = a.suite
    a.outdir = a.outdir or SUITE_DIRS[SUITE]
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 1)))
    {"pilot": cmd_pilot, "list": cmd_list, "run": cmd_run, "summarize": cmd_summarize}[a.command](a)
