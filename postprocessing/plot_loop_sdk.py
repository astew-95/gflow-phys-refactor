"""Advisor-facing figures for the 'large diamond + K small diamonds' sweep (HANDOFF step 4f).

Compares ld_sd{2,8,32}_free (outputs_loop_sdk/) with ld_free (K=0) and the ld_tree anchor
(outputs_loop_big/), 10x10 periodic Ising, 12 temperatures x 3 seeds.
Writes figures/loop_sdk/{sdk_accuracy,sdk_effect_fit,sdk_loss,sdk_observables}.{png,pdf}
and prints the fit numbers.
Usage: python plot_loop_sdk.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures", "loop_sdk")
TC = 2.0 / np.log(1 + np.sqrt(2))

# K = small diamonds per pair of sampled trajectories. Marker shape is a second encoding,
# since the K lines lie on top of each other.
KS = [  # (K, config, label, color, marker)
    (0,  "ld_free",      "whole-path check only (baseline)",          "#eb6834", "o"),
    (2,  "ld_sd2_free",  "+ 2 local swap checks",                      "#2a78d6", "s"),
    (8,  "ld_sd8_free",  "+ 8 local swap checks",                      "#1baf7a", "^"),
    (32, "ld_sd32_free", "+ 32 local swap checks",                     "#4a3aa7", "D"),
]
TREE = ("ld_tree", "fixed site order (reference)", "#8a8984")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 200,
})


def load():
    runs = defaultdict(lambda: defaultdict(list))  # config -> T -> [json]
    wolff, exact = {}, {}
    files = glob.glob(os.path.join(HERE, "outputs_loop_sdk", "*.json"))
    files += glob.glob(os.path.join(HERE, "outputs_loop_big", "*.json"))
    for f in files:
        d = json.load(open(f))
        T = round(d["T"], 3)
        exact[T] = d["exact"]
        if d["kind"] == "mcmc":
            wolff[T] = d["final"]
        elif d["mult"] == 1 and d["config"] in {c for _, c, *_ in KS} | {TREE[0]}:
            runs[d["config"]][T].append(d)
    return runs, wolff, exact


def stat(runs_T, f):
    Ts = sorted(runs_T)
    v = [np.array([f(d) for d in runs_T[T]]) for T in Ts]
    return np.array(Ts), np.array([x.mean() for x in v]), np.array([x.min() for x in v]), \
        np.array([x.max() for x in v])


def tc_line(ax, label=True):
    ax.axvline(TC, color=INK2, lw=0.8, ls=":", zorder=1)
    if label:
        ax.text(TC + 0.02, 0.97, "critical\ntemperature", transform=ax.get_xaxis_transform(),
                va="top", fontsize=8, color=INK2)


def draw_all(ax, runs, f, tree=True):
    if tree:
        Ts, mu, lo, hi = stat(runs[TREE[0]], f)
        ax.plot(Ts, mu, "--", color=TREE[2], lw=1.5, zorder=2)
    for _, cfg, _, col, mk in KS:
        Ts, mu, lo, hi = stat(runs[cfg], f)
        ax.plot(Ts, mu, "-", color=col, marker=mk, zorder=3, lw=1.5,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.fill_between(Ts, lo, hi, color=col, alpha=0.10, lw=0, zorder=1)


def handles(tree=True, extra=()):
    h = [Line2D([], [], color=col, marker=mk, label=lab, markeredgecolor="white")
         for _, _, lab, col, mk in KS]
    if tree:
        h.append(Line2D([], [], color=TREE[2], ls="--", label=TREE[1]))
    return h + list(extra)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def fig_accuracy(runs):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    draw_all(ax, runs, lambda d: d["final"]["kl_q_p"])
    tc_line(ax)
    ax.set_xlabel("temperature T")
    ax.set_ylabel("error of learned sampler\n(KL divergence to true distribution, nats)")
    ax.set_title("Adding local swap checks does not reduce sampler error (10x10 Ising, random order)",
                 fontsize=10, loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(handles=handles(), fontsize=8, loc="upper left")
    fig.text(0.0, -0.04, "Lines: mean of 3 training seeds; shaded: min-max. Lower is better.",
             fontsize=8, color=INK2)
    save(fig, "sdk_accuracy")


def l1_fit(x, y):
    """Least-absolute-deviation line y = a + b x (robust to the few outlier seeds)."""
    X = np.column_stack([np.ones_like(x), x])
    p = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(200):  # IRLS
        w = 1 / np.maximum(np.abs(y - X @ p), 1e-6)
        p = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    return p


def fig_effect_fit(runs):
    """Per-seed % change in KL vs the K=0 baseline mean at the same T, against log2(1+K)."""
    base = {T: np.mean([d["final"]["kl_q_p"] for d in runs["ld_free"][T]]) for T in runs["ld_free"]}
    xs, ys, Tcol = [], [], []
    for K, cfg, *_ in KS:
        for T, ds in runs[cfg].items():
            for d in ds:
                xs.append(np.log2(1 + K))
                ys.append(100 * (d["final"]["kl_q_p"] / base[T] - 1))
                Tcol.append(T)
    xs, ys, Tcol = map(np.array, (xs, ys, Tcol))
    a, b = l1_fit(xs, ys)
    # Bootstrap over temperatures (seeds at one T share the baseline, so resample T as a block).
    rng = np.random.default_rng(0)
    Tu = np.unique(Tcol)
    boots = []
    for _ in range(2000):
        pick = rng.choice(Tu, len(Tu))
        idx = np.concatenate([np.where(Tcol == t)[0] for t in pick])
        boots.append(l1_fit(xs[idx], ys[idx]))
    boots = np.array(boots)
    blo, bhi = np.percentile(boots[:, 1], [2.5, 97.5])
    med = {K: np.median(ys[xs == np.log2(1 + K)]) for K, *_ in KS}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    jit = np.random.default_rng(1).uniform(-0.12, 0.12, len(xs))
    for K, _, _, col, mk in KS:
        m = xs == np.log2(1 + K)
        ax.scatter(xs[m] + jit[m], ys[m], s=22, color=col, marker=mk, alpha=0.55, lw=0, zorder=3)
    xg = np.linspace(0, np.log2(33), 50)
    band = np.array([bb[0] + bb[1] * xg for bb in boots])
    ax.fill_between(xg, *np.percentile(band, [2.5, 97.5], axis=0), color=INK2, alpha=0.15, lw=0)
    ax.plot(xg, a + b * xg, color=INK, lw=2, zorder=4)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([np.log2(1 + K) for K, *_ in KS])
    ax.set_xticklabels([str(K) for K, *_ in KS])
    ax.set_xlabel("local swap checks added per pair of sampled paths")
    ax.set_ylabel("change in sampler error vs baseline (%)\n(negative = better)")
    ax.set_title(f"Robust (L1) trend: {b:+.2f}% per doubling  [95% CI {blo:+.2f}, {bhi:+.2f}]",
                 fontsize=10, loc="left")
    fig.text(0.0, -0.06, "Each dot = one training seed at one of 12 temperatures. Line: least-absolute-"
             "deviation fit; band: 95% bootstrap over temperatures.", fontsize=8, color=INK2)
    save(fig, "sdk_effect_fit")
    print(f"L1 fit: intercept {a:+.2f}%, slope {b:+.3f}% per doubling of (1+K), "
          f"95% CI [{blo:+.3f}, {bhi:+.3f}]")
    print("median % change by K:", {K: round(v, 2) for K, v in med.items()})
    # Wins: K>0 seed-mean beats baseline mean at each T.
    for K, cfg, *_ in KS[1:]:
        wins = sum(np.mean([d["final"]["kl_q_p"] for d in runs[cfg][T]]) < base[T] for T in base)
        print(f"K={K}: lower mean KL than baseline at {wins}/{len(base)} temperatures")


def fig_loss(runs):
    Tshow = [1.5, 2.27, 3.0]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, T in zip(axs, Tshow):
        for _, cfg, _, col, mk in KS:
            ds = runs[cfg][T]
            it = np.array([c["iters"] for c in ds[0]["curve"]])
            L = np.array([[c["loss"] for c in d["curve"]] for d in ds])
            ax.plot(it, L.mean(0), color=col, marker=mk, markevery=4, lw=1.5,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.fill_between(it, L.min(0), L.max(0), color=col, alpha=0.10, lw=0)
        ax.set_yscale("log")
        ax.set_title(f"T = {T}" + ("  (critical)" if T == 2.27 else ""), fontsize=10, loc="left")
        ax.set_xlabel("training iteration")
    axs[0].set_ylabel("training loss (log scale)")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.legend(handles=handles(tree=False), fontsize=8, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0))
    fig.text(0.0, -0.06, "Total loss incl. swap-check term (~0.001, <0.1% of the total). The rise-then-drop "
             "shape comes from the temperature annealing schedule shared by all runs, not from the objectives. "
             "Mean of 3 seeds, shaded min-max.", fontsize=8, color=INK2)
    save(fig, "sdk_loss")


def fig_observables(runs, wolff, exact):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    Ts = sorted(exact)
    ax = axs[0]
    draw_all(ax, runs, lambda d: d["final"]["E_per_spin"])
    ax.plot(Ts, [exact[T]["E_per_spin"] for T in Ts], color=INK, lw=1.2, zorder=4)
    ax.set_ylabel("energy per spin")
    ax.set_title("Average energy", fontsize=10, loc="left")
    ax = axs[1]
    draw_all(ax, runs, lambda d: d["final"]["abs_m"])
    Tw = sorted(wolff)
    ax.plot(Tw, [wolff[T]["abs_m"] for T in Tw], color=INK, lw=1.2, zorder=4)
    ax.set_ylabel("magnetization |m|")
    ax.set_title("Average magnetization", fontsize=10, loc="left")
    for ax in axs:
        tc_line(ax, label=False)
        ax.set_xlabel("temperature T")
    ref = Line2D([], [], color=INK, lw=1.2, label="ground truth (exact / long Monte Carlo)")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.legend(handles=handles(extra=[ref]), fontsize=8, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "sdk_observables")


if __name__ == "__main__":
    runs, wolff, exact = load()
    fig_accuracy(runs)
    fig_effect_fit(runs)
    fig_loss(runs)
    fig_observables(runs, wolff, exact)
    print("wrote", OUT)
