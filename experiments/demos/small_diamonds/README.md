# Small diamonds in the KC / loop GFlowNet: code and evidence

This bundle has everything needed to check why adding small diamonds (two-step swaps) to the
large-diamond (KC) loss does not improve the sampler. It also covers why local-only losses
(small diamonds + one-step kites) fail, and why upweighting the terminal kites makes them worse.
A plain-language summary of the results is in the overview page that was shared with this bundle.

## Start here: a 10-second exact demo

```
cd code
python demo_small_diamonds.py          # needs torch + numpy; output also in results/demo_output.txt
```

The demo works on the 3x3 periodic Ising model at T = 2.27 with the free-order DAG. Every loss is
an exact expectation over all 3^9 partial states, with no training and no sampling noise.

| policy | small-diamond loss | KL(q‖p) |
|---|---|---|
| exact optimum for T = 2.27 | 5e-31 | 0.000 |
| **uniform random policy** | **0 exactly** | **2.78** |
| **exact optimum for T = 1.0** | **2e-30** | **0.38** |
| **exact optimum for T = 5.0** | **1e-31** | **1.18** |
| optimum + 0.02 logit noise | 7e-4 | 0.000 (but log P_F(x) varies 0.05 nats across orders) |

Any order-consistent policy satisfies every small diamond, including the uniform policy and the
exact sampler for the wrong temperature. The square's two paths share both endpoints, so log R
cancels (and with uniform P_B, so does P_B). The loss has no access to the energy. The last row
gives a sense of scale: the trained 10x10 models end with a small-diamond loss of 4e-4 to 1e-3,
about the size that 0.02 nats of random logit noise produces.

## Where the losses are in the code

All line numbers refer to the files in `code/`.

**Training losses (sampled, used for the 10x10 runs): `loop_gfn.py`**
- `LoopGFlowNet.loss`, line 278. The base batch is rolled out once (`_rollout`, line 196), with
  epsilon-uniform exploration at line 214, and replayed with gradients (`_replay`, line 233).
- Large diamond (KC), lines 288-292: `V = sum log P_F - log R`; loss = `Var(V)` over the batch.
  This is the only term that sees log R for whole configurations.
- Small diamond, lines 331-358: for position t of a base trajectory with actions (a, b), compare
  `lp(a|s) + lp(b|s+a)` (already in the replay) with `lp(b|s) + lp(a|s+b)` (two extra policy
  evaluations, `_prefix_state` at line 177 rebuilds s). There is no reward anywhere in this block.
  Note line 356: `l_sd = 0.5 * mean(...)`. This is a **mean** over diamonds, so in the K = 2/8/32
  sweep K only reduced the noise of the term and never increased its weight. We fixed this
  afterwards by sweeping `w_sd` up to 1000 (results below).
- One-step kite, the `kite_all` branch and the `kite_lo = kite_hi = 1` case: the local-field check
  `[lp(a|s) - lp(a^1|s)] = log R(x) - log R(x with spin flipped)`. This is the only local term
  with the energy in it.
- The free-order policy (`FreeOrderPolicy`, line 74) is one MLP that outputs 2D logits (site,
  value) from (spins, mask). All states share its parameters, which matters for the next point.

**Exact engine (no sampling, optionally no network): `kc_exact.py`**
- `ExactDAG._losses`, line 172: ld (line ~175), kite (line 194) and sd (line 201) as exact
  expectations under the policy's own visitation weights (detached, like training).
- `optimal_theta`, line 83: the exact optimal free-DAG policy.
- `method_weights`, line 223: the weighted variants used in the advisor tests
  (`basis_kw<w>` = w·kite + sd, `ld_sdw<w>` = ld + w·sd, `ld_basis` = ld + kite + sd).
- `test_loop_gfn.py` checks every loss is 0 at the exact optimum and > 0 when perturbed.

## The evidence, in order

1. **The small diamonds add nothing at 10x10** (`results/summary_sdk.txt`, `figures/sdk_*.png`,
   `results/effect_fit.txt`). With K = 0/2/8/32 per pair, 12 temperatures and 3 seeds, the KL curves
   overlap. A robust L1 fit gives +0.05% KL per doubling of K, 95% CI [-0.13, +0.19].
2. **LD alone already satisfies them where it samples** (`results/diagnose_sdk.json` vs
   `results/diagnose_main.json`, produced by `diagnose_loops.py`). On-policy small-diamond rms is
   0.020-0.022 (T = 1.5) and 0.035 (T = 2.27), the same with or without the term. The order spread
   of log P_F is the same too.
3. **The real errors are off the sampler's support** (same files, key `p` = Wolff samples from the
   target). LD-free has V spread 0.7 on its own samples but 32 on target samples at T = 1.5 (mode
   collapse), and its kite residual is biased by -1.3 nats there. Diamonds built from q's
   trajectories never reach those states.
4. **Local-only losses fail because of the network, not the objective** (`results/summary_exact_ladder.txt`,
   SK D = 6/8/10). With a lookup table and exact gradients, basis (kite + sd) reaches KL ~1e-12.
   With an MLP and exact gradients it is 100-1000x worse than ld, and the gap grows with D. The
   shared network leaves a small residual on every local check, and those residuals add up over
   the ~D²/4 swaps between orders (step 3 of the demo, at scale).
5. **Weighting tests** (`results/summary_kitew.txt`; columns T = 2.27/1.5 are 3x3 Ising,
   T = 1.0/0.4 are SK D = 10; `results/summary_sdw.txt` is 10x10):
   - Upweighting terminal kites in basis is monotonically **worse**: Ising T = 2.27 KL is
     7.4e-3 at 1x, 1.3e-2 at 10x and 9.0e-2 at 100x. **Down**weighting helps (8e-4 at 0.1x) but stays
     70x behind ld (1.2e-5).
   - Weighting small diamonds 10/100/1000x next to ld: at 10x10 the sd residual drops ~4x but KL is
     unchanged (T = 2.27: 2.11 vs 2.10/2.09/2.10). On small systems, more weight is monotonically worse.
   - ld + kite + sd: slightly worse than ld with exact gradients. With sampled gradients it is
     1.2-2.4x better in 3 of 4 cells, but uses ~40% more compute per step and is not
     compute-matched. This is still open.
6. **Fixed order beats random order 3-6x near and above T_c** (`results/summary_main_10x10.txt`:
   `ld_tree` vs `ld_free`), whatever loss terms are added.

## Reproducing

The environment is Python 3.10 with torch, numpy and matplotlib (scipy is not needed).
- Tests: `python test_loop_gfn.py`.
- Tables: `python loop_big.py summarize --suite sdk` (or `sdw`, or no flag for the main suite),
  `python summarize_kc_ladder.py`, `python summarize_kc_ising3.py kitew`. Plots:
  `python plot_loop_sdk.py`. All of these read the result JSONs in `code/outputs_*`.
- Rerunning training: `python loop_big.py run --suite sdk --task i` (one 10x10 task, ~40 min on
  2 cores) and `python kc_exact.py grid-run --grid kitew --task i`. The SLURM array scripts used
  are in `slurm/`. Trained model weights are not included.
- `lab_notes.md` is the running research log (terse; the numbered steps match the history above).
