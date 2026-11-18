"""
demo_small_diamonds.py
======================
Why small diamonds can't train a sampler on their own, shown exactly (no training,
no sampling noise) on the 3x3 periodic Ising model at T = 2.27, free-order DAG.

Uses kc_exact.ExactDAG, which enumerates all 3^9 partial states and computes every
loss as an exact expectation under the policy's own trajectory distribution.

  ld    large diamond  = Var over trajectories of log P_F(tau) - log P_B(tau|x) - log R(x)
  kite  one-step kite  = local-field check at the last step (sees the energy)
  sd    small diamond  = swap two consecutive actions; same endpoints, energy cancels

Policies compared:
  A  exact optimal policy for T = 2.27                  -> everything ~0, KL ~0
  B  uniform random policy (all logits 0)               -> sd = 0 EXACTLY, KL large
  C  exact optimal policy for a DIFFERENT T (1.0, 5.0)  -> sd = 0 EXACTLY, KL large
  D  policy A + iid noise on every logit                -> small sd, but log P_F(x)
                                                           differs across site orders
B and C pass every small diamond, yet are wrong: small diamonds enforce consistency
of P_F across orders, and any consistent sampler (including wrong ones) satisfies them.
D shows the other half: tiny local (per-swap) errors add up across orderings.

Usage: python demo_small_diamonds.py      (~10 s on a laptop CPU)
"""
import math
import torch
from theory.kc_exact import ExactDAG
torch.manual_seed(0)
D, T = 9, 2.27


def order_spread(g, theta, n_x=200, n_orders=32):
    """Mean over x ~ target of the std (across random site orders) of log P_F(tau | x)."""
    lp = g.logp(theta)
    pw = 3 ** torch.arange(D)
    term = g.by_level[D]
    p = g.target_q()
    xs = term[torch.multinomial(p, n_x, replacement=True)]
    digits = (xs.unsqueeze(1) // pw) % 3                      # 1 = spin -1, 2 = spin +1
    stds = []
    for dig in digits:
        sums = []
        for _ in range(n_orders):
            code, tot = 0, 0.0
            for site in torch.randperm(D).tolist():
                v = int(dig[site])
                tot += float(lp[code, 2 * site + (v - 1)])
                code += int(pw[site]) * v
            sums.append(tot)
        stds.append(torch.tensor(sums).std().item())
    return sum(stds) / len(stds)


def report(name, g, theta):
    L = {k: float(v) for k, v in g.losses(theta, ("ld", "kite", "sd")).items()}
    m = g.metrics(theta)
    print(f"{name:38s} sd={L['sd']:9.2e}  kite={L['kite']:9.2e}  ld={L['ld']:9.2e}  "
          f"KL={m['kl_qp']:7.4f}  TV={m['tv']:.3f}  order_spread={order_spread(g, theta):.3f}")


g = ExactDAG(D, T, dag="free", kind="ising")
opt = g.optimal_theta()
print(f"3x3 periodic Ising, T = {T}, free-order DAG. Losses are exact expectations.\n")
report("A  optimal policy (T=2.27)", g, opt)
report("B  uniform random policy", g, torch.zeros_like(opt))
for T_other in (1.0, 5.0):
    report(f"C  optimal policy for T={T_other}", g, ExactDAG(D, T_other, dag="free", kind="ising").optimal_theta())
for sigma in (0.02, 0.05, 0.1):
    noisy = opt + sigma * torch.randn_like(opt)
    report(f"D  optimal + logit noise sigma={sigma}", g, noisy)
print("""
Read across: B and C have sd = 0 (to float precision) while being far from the target,
so the small-diamond loss cannot tell a right sampler from a wrong one. Only ld and the
kite involve the energy. In D, sd stays small while log P_F(x) still depends on the
site order; that accumulated path dependence is what grows with system size (D = 100 needs
~D^2/4 = 2500 swaps to go between two random orders).""")
