INTERPRETATION
This screen changes clipping threshold and learning rate, keeping the local loss SD+.1*kite
fixed. The supplied runs document LR/weight sweeps and norm-1 clipping, but not this
clipping-by-objective screen. This is new relative to those records, not a claim of literature novelty.
Question: does changing clipping help local and global objectives differently?
Compare clip=off/.1/1 WITHIN each method and LR, paired by seed. Then compare the change
between methods. Select by median final KL or time to a prespecified KL threshold, not best seed.
A local-specific benefit would motivate a fresh-seed confirmation; no benefit is also informative.
Do not assume clipping is bad, or promise that either method will win.

Each worker gets the same wall-time slot (including imports/setup/evaluation), so update counts
can differ. Curves separately record active optimizer time, elapsed time, KL in both directions,
TV, preclip gradient norms and clipped-step fractions. Diagnostics/initialization are excluded
from active optimizer time. Peak RSS includes Python/PyTorch/data, not just model tensors.
First-threshold crossings are approximate and sampled about every five seconds, not guaranteed
sustained convergence. Read whole curves; empty threshold fields mean not reached by budget.
The memory/seconds recorded here characterize this exact-enumeration implementation.
They do NOT establish that sampled local training is cheaper at D=100. That needs a sampled test.

Three seeds, two LRs and a short budget are exploratory. This does not exhaust the tuning space,
and it may not reach the KL values in longer historical runs. Constant LR intentionally avoids
stretching a cosine schedule across unequal update counts. Checkpoints are result curves only,
not optimizer-state checkpoints. Source modules are never overwritten.
Use the same source/dependencies/hardware for comparisons. Execution is CPU-only, two threads
by default, no entropy bonus or beta annealing. On-policy expectation weights stay detached.
After a pilot, confirm the selected settings against the SAME-LR norm-1 baseline on fresh seeds.
This script was syntax-checked and its configuration/summary logic checked here; numerical
training was not run here because PyTorch and loop_gfn.py were unavailable.
