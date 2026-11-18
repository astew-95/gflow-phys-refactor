# Handoff: loop-family log Z-free GFlowNet objectives

## Context
- Project: KC (Kolmogorov Criterion) GFlowNet on spin glasses (2D Ising, SK).
- KC (= full-batch variance of V(tau) = log P_F(tau) - beta*log R(x)) turned out to be
  previously published: Zhang et al. 2023 "Robust Scheduling with GFlowNets" (log-partition
  variance loss) and da Silva et al. 2024 "Embarrassingly Parallel GFlowNets" (contrastive balance).
- Pivot: extend it with a family of loop objectives, all of the form 1/2 (V(tau) - V(tau'))^2:
  - Large diamond: two independent trajectories (= existing KC). Tree and free DAG.
  - Kite: partner shares the first t_b actions, takes a different action at t_b, then rolls out
    independently. Shared tail cancels. t_b=0 -> large diamond; t_b=D-1 -> one-step local-field
    constraint. Prior art: Malkin et al. 2022 (TB paper) App. A.2 "non-forward trajectories";
    sampling resembles Local Search GFlowNets (Kim et al. ICLR 2024). Novelty here is empirical
    (branch depth as a knob), not the constraint.
  - Small diamond: swap two consecutive actions (same values). Free-order DAG only. Reward and
    uniform P_B cancel, so it only enforces order-consistency of P_F (it never sees the energy).
    Not found as an explicit objective in the literature; implied by DB/SubTB.
- Also check: Delta-AI (Falet et al. ICLR 2024), close to the single-spin-flip partner idea.
- Cycle-basis conjecture: on the free DAG, small diamonds + one-step kites generate the full
  cycle space (incl. sink), so enforcing KC on just those suffices in the exact limit. On the
  tree, one-step kites alone should NOT suffice (expected negative control).

## Files
- `spin_glass_lib.py` - unchanged (SpinEnv, StepEmbedPolicy, BatchedGFlowNetBase, TB, KC, MCMC).
- `loop_gfn.py` - NEW. `LoopGFlowNet` with `dag="tree"` (default, fixed site order, current setup)
  or `dag="free"` (any unassigned site next, action = 2*site + value, uniform P_B which cancels
  from every loss). Weights w_ld / w_kite / w_sd, kite depth range [kite_lo, kite_hi]*(D-1),
  `kite_diverge` = flip (same site, opposite value) or any. Presets in `LOOP_PRESETS`.
- `run_tc_sweep.py` - PATCHED. New method `LOOP`; flags --dag, --loop_preset, --w_ld, --w_kite,
  --w_sd, --kite_lo, --kite_hi, --n_kite, --n_sd, --kite_diverge, --tag_suffix (all in the
  checkpoint signature). w_sd > 0 with --dag tree is rejected.
- `test_loop_gfn.py` - NEW. Builds the exact optimal policy by enumeration (SK, D=5) for both DAGs;
  asserts each loss ~0 at the exact policy, >0 when perturbed; checks exact-policy sampling
  matches Boltzmann (TV < 0.03).

## Status: test_loop_gfn.py ALL PASS (2026-10-08)
Fixed: `_prefix_state` dtype mismatch under float64; NaN grads from entropy on free DAG
(p * -inf for illegal actions). Test now also checks grads stay finite over training steps.
Next steps, in order:
1. (done) `python test_loop_gfn.py`.
2. (done 2026-10-08) Regression: `LOOP --loop_preset ld --dag tree` should reproduce KC's loss on identical actions.
   Verified: same seed + weights -> identical actions; over 50 lockstep Adam steps (SK D=10,
   Ising D=16) max |dLoss| < 1e-6, |dGrad| < 2e-6, |dParam| < 2e-5 (float32 roundoff).
3. (done 2026-10-08, job 1036033) Quick SK D=8 run vs exact enumeration (TV distance): presets ld, kite, kite_last (tree) and
   basis (free). Sanity check only, not paper results.
   loop_d8_exact_check.py + submit_loop_d8_exact.slurm -> outputs_loop_d8/. EXACT TV (mass pushed
   through all partial states), 4000 iters, batch 16->64, T in {1.0, 0.4}, J_seed {0,1}:
     TB_tree 0.007-0.012 | ld_tree 0.009-0.023 | kite_tree 0.004-0.007 (best)
     kite_last_tree 0.55-0.94 (fails, as predicted: low loss but wrong distribution)
     ld_free 0.004-0.020 | basis_free 0.035-0.092 (converges, but clearly slower/worse)
   Open: is basis_free's gap just slow propagation through one-step constraints (try more iters)?
3b. (done 2026-10-08, job 1036078) COMPUTE-MATCHED GRID. loop_grid.py + submit_loop_grid.slurm ->
   outputs_loop_grid/ (496 jsons); plots/table: `python make_plots.py loop_grid` -> figures/loop_grid/.
   Instrumentation: BatchedGFlowNetBase._policy_fwd counts policy rows (evals_grad / evals_nograd);
   no loss math changed (test_loop_gfn.py + KC regression still pass). Budget = kite_tree evals_grad
   in 4000 iters, B 16->64 (D6/8/10: 1.92M/2.56M/3.20M); schedules scaled to each run's iteration
   count (anneal_frac 0.999-1.001). 8 seed combos (4 J x 2 train) per cell; exact TV.
   Findings (differences quoted only where they beat the paired seed spread):
   - Tree, matched compute: TB, ld, ld2x, kite, ld_kite are INDISTINGUISHABLE (paired diffs within
     +-1 std, e.g. D10 T0.4 kite-ld = -0.002+-0.003). Step 3's "kite best" was a 2x-compute artifact.
   - Free DAG: kite is consistently a bit worse than ld (0/8 wins in 3 of 4 cells, +0.004..0.013 TV).
     kite_sd vs kite: no consistent effect. Free DAG is never better than tree at matched compute.
   - Depth bands (tree D10): early (0.016/0.019 at T=1/0.4) == full (0.016/0.020). mid (0.22/0.45) and
     late (0.46/0.79) FAIL and plateau (flat TV over last 50% of budget) with LOW loss (late ~1e-3):
     with t_b >= t_min the first t_min decisions get no gradient (shared prefix cancels), so
     root-level marginals are never constrained. Structural, not slow convergence.
   - basis_free vs ld_free gap: grows sharply with D (T=1 x1: 0.002/0.027/0.100 at D6/8/10;
     T=0.4: 0.014/0.067/0.41) and shrinks with 4x compute (T=1 D10 0.100->0.033; T=0.4 D10
     0.41->0.093) -> mostly slow propagation, getting worse with D. But at T=0.4 D=6 the gap
     (~0.015-0.02) does NOT shrink with compute, and the T=0.4 x4 curves flatten at the end, so a
     small residual non-propagation component can't be ruled out. basis is not viable at N=100.
   - Loss magnitudes are not comparable across DAGs (free ld loss ~0.1 at TV 0.03 vs tree ~0.006).
   - Caveat: TV-vs-compute curves are dominated by the beta anneal (TV is vs the beta=1 target),
     so only endpoints compare objectives; a fixed-beta run would give true convergence curves.
4. (RUNNING, submitted 2026-10-08, job 1036717) BIG RUNS: 10x10 periodic Ising, FREE (random-order)
   DAG focus. loop_big.py + submit_loop_big.slurm -> outputs_loop_big/ (one json per task + .policy.pt).
   264 tasks: [ld_tree (anchor), ld_free, kite_free, ld_sd_free, basis_free] x 12 T (1.5..3.2, dense
   near Tc=2.269) x 3 seeds at budget x1 (= ld_tree evals_grad over 8000 iters, B 16->64; 32.0M);
   12 long Wolff reference runs; basis_free + ld_free at budget x4 (propagation check).
   Reference: EXACT Kaufman finite-lattice solution (ising_exact.py; validated vs 4x4 brute force):
   log Z, <E>, C. Metrics per task: KL(q||p), IS log Z error, ESS (log q exact on tree; K=16 random-
   order IS estimate on free DAG, validated), E/D, C/D, |m|, chi, U4 (vs Wolff), frac(m>0).
   Pilot (job 1036709): 0.4-0.6 h/task at x1 on 2 cores; untrained KL ~24 nats.
   Fault tolerance: checkpoint every 5% of budget + resume; --requeue for preemption; USR1 trap
   requeues 5 min before the 3 h limit (tested with job 1036715 -> logs/loop_big_rqtest_*).
   Resubmitting submit_loop_big.slurm is always safe (done tasks skip, partial ones resume).
   Check: `ls outputs_loop_big/*.json | wc -l` (264 = done); `python loop_big.py summarize`.
   Rationale for configs: ld_sd = "do small diamonds help KC"; basis = DB-like end (one-step kites
   are the terminal edges, small diamonds the interior constraints) -> expect slow propagation.
   RESULTS (all 264 COMPLETED; anneal_frac >= 0.999; Wolff E/D within 0.003 of exact):
   KL(q||p) nats, mean over 3 seeds, at T = 1.5 / 2.0 / 2.27 / 2.6 / 3.2:
     ld_tree      0.71 / 0.23 / 0.46 / 0.72 / 0.57    (ESS 0.95 / 0.39 / 0.25 / 0.20 / 0.31)
     ld_free      0.78 / 1.28 / 2.11 / 3.64 / 3.18    (ESS ~0 for T >= 2.27)
     ld_free x4   0.74 / 1.05 / 1.59 / 2.61 / 2.01
     ld_sd_free   0.79 / 1.31 / 2.12 / 3.79 / 3.13    (== ld_free within seed spread)
     kite_free    0.75 / 1.41 / 2.40 / 6.53 / 4.59    (worse than ld_free above Tc)
     basis_free   40.7 / 19.5 / 13.3 / 8.3 / 5.0     x4: 57.4 / 29.0 / 19.6 / 13.6 / 8.4 (WORSE)
   (free-DAG KL is an IS estimate biased LOW, so free configs are at least this bad.)
   - The tree anchor is far better than every free-DAG config at matched compute (3-6x lower KL
     near/above Tc). All free objectives (ld, ld_sd, kite) fail similarly and x4 compute helps
     only modestly -> the bottleneck looks like the free DAG/FreeOrderPolicy, not the loss.
     Free samplers are too ordered for T <= 2.6 (|m| +0.1..0.36 vs Wolff, chi ~0.1x) and too
     disordered for T >= 2.8 (E/D +0.33..0.6).
   - Small diamonds add nothing measurable to KC (ld_sd == ld_free).
   - basis_free fails outright: near-zero |m| at all T (E/D off by +0.5..1.5), and x4 compute makes
     it WORSE while its loss goes DOWN (lowest loss of all free configs). Not slow propagation at
     D=100; plausibly self-consistent on its own (disordered) support, which it never leaves.
   - Mode collapse is visible exactly: ld_tree at T <= 1.8 puts each seed in ONE mode (frac m>0 =
     0 or 1) and KL ~ 0.71-0.76 ~ log 2 + small. Free configs collapse up to T ~ 2.45.
   - ld_tree itself: E/D within 0.07 of exact, chi 0.6-1.1x Wolff, slightly over-ordered near Tc.
4b. (done 2026-10-14) WHY basis_free FAILS. diagnose_loops.py -> figures/loop_big/diagnose_loops.json.
   Loads seed-0 policies, measures every residual at beta=1 on q-rollouts and on Wolff samples
   (random orders). Plots of step 4: plot_loop_big.py -> figures/loop_big/.
   - Theory is fine (zero on basis => zero everywhere), but it says nothing about NEAR-zero.
     Small diamonds are tiny (rms 0.04-0.09 nats, mean ~0, random sign), yet log P_F(tau|x)
     varies by 1.3-4 nats across site orders for the SAME x (order_std). Matches random-walk
     accumulation over the ~D^2/4 ~ 2500 swaps between two random orders: 50 * rms ~ 2-4 nats.
     Global error ~ D * local error -> explains why the basis gap grows sharply with D.
   - The only energy signal (last-step kite) is poorly fit: rms 0.8-1.7 nats (ld_tree 0.2-0.3),
     biased underconfident on target samples (mean -0.7..-1.0) -> too-soft local fields ->
     disordered sampler. x4 compute barely moves it (1.69->1.52 at T=1.5).
   - Diamonds carry no energy info: any consistent joint q satisfies them (incl. a near-product
     disordered one). Loss falls with x4 because on-policy kite rms falls (on q's disordered
     states) while target-sample kite rms RISES (T=1.5: 1.65->1.90) -> KL worse.
   - ld_free also has order_std 0.8-1.6 nats, so the K=16 IS log q is biased low by O(sigma^2/2):
     free-DAG KLs are underestimates, possibly by several nats.

4c. (done 2026-10-14, job 1041630) IS THE BASIS FAILURE FUNDAMENTAL? kc_exact.py = exact tabular engine
   (all 3^D free-DAG states, every loss an exact expectation, detached sampling weights like training;
   validated: 0 at optimum, ld matches MC). Ladder {table, net} x {exact, sampled grads} x {ld, basis,
   basis w/ uniform "perfect coverage" weights}, SK D=6/8/10, T=1/0.4, J=0/1, lr tuned + cosine.
   outputs_kc_ladder/, table: python summarize_kc_ladder.py. Final exact KL, D=10 (T=1 | T=0.4):
     table/exact    ld 6e-11 | 3e-5     basis 1e-12 | 1e-5   basis_unif 1e-12 | 1e-12
     net/exact      ld 2e-5  | 2e-4     basis 9e-3  | 0.25   basis_unif 2e-3  | 0.077
     net/sampled    ld 1e-3  | 0.016    basis 0.11  | 1.7
   - Ideal optimizer + lookup table: basis is PERFECT (no intrinsic conditioning problem).
   - Network alone (exact grads, no sampling noise) breaks it: basis 100-1000x worse than ld, gap
     grows with D. Perfect coverage helps but does not close it at D=10 (0.077 vs 2e-4 at T=0.4).
     -> shared function approximation leaves a local-residual floor; global error accumulates over
     many local cycles (step 4b), while ld penalizes the global error directly.
   - Sampling noise hurts both, basis more. LR tuning/cosine annealing was in every cell.
   Round 2 (job 1041841, tasks 204-299): dense basis sampling (sd_all/kite_all in loop_gfn.py,
   B=16 compute-matched and B=64) and a 4x wider net (net512) under exact grads.

4d. (2026-10-20) 3x3 PERIODIC ISING (D=9) in kc_exact.py (--kind ising; J = init seed there).
   Also added replay-buffer loss weightings (target = backward replay of x~p, selfbuf = of x~q,
   mix_* = 50/50 with on-policy). Grid `--grid ising3` (job 1045547) was CANCELLED part-way at
   the user's request (42/120 in outputs_kc_ising3/). Early: target-only buffer basis -> low loss but
   TV 0.90 (off-target mass unconstrained); net/sampled basis T=2.27 KL 1.3e-2 vs ld 1.7e-3.
4f. (DONE 2026-10-21, job 1045830, 108/108 COMPLETED) HOW MANY SMALL DIAMONDS PER LARGE DIAMOND, 10x10 Ising.
   `python loop_big.py run --suite sdk` -> outputs_loop_sdk/ (108 tasks; submit_loop_sdk.slurm).
   ld_sd{2,8,32}_free = ld + K small diamonds per large diamond (trajectory pair), built from the
   ld batch: n_sd = K/2 distinct positions per base trajectory (new no-replacement path in
   LoopGFlowNet when 1 < n_sd <= D-1; n_sd=1 path unchanged). ITERATION-matched to ld_free (8000 its,
   same schedules); extra evals_grad +2/8/32% recorded, not charged. K=0 reference = main-suite
   ld_free (8001 its). Old ld_sd_free (step 4) was compute-matched (7844 its) -> K=2 rerun.
   Table: `python loop_big.py summarize --suite sdk`. Same T grid / seeds / metrics as step 4.
   RESULT: NO EFFECT. KL(q||p) for K=0/2/8/32 overlaps at all 12 T (within seed spread);
   robust L1 fit of per-seed % KL change vs log2(1+K): +0.05% per doubling, 95% CI [-0.13, +0.19]
   (bootstrap over T). K>0 beats K=0 mean at 7/6/5 of 12 T (coin flip). sd loss term ~1e-3 vs ld ~3.
   Tree anchor still 3-6x lower KL near/above Tc. Figures: python plot_loop_sdk.py -> figures/loop_sdk/.
4g. (DONE 2026-10-21, all 295 COMPLETED) FOLLOW-UPS TO 4f + ADVISOR'S SUGGESTIONS.
   Flaw in 4f: l_sd is a MEAN over diamonds, so K only changed the noise of the sd term, never its
   weight (sd ~1e-3 vs ld ~3 at w_sd=1). Also measured (diagnose_loops.py, now --outputs/--match/--out;
   job 1046071 -> figures/loop_sdk/diagnose_sdk.json): sd rms and order_std of ld_sd{2,8,32} == ld_free
   (T1.5: sd rms 0.020 vs 0.022, order_std 0.54-0.62 vs 0.56; T2.27: 0.035 vs 0.035, 0.84 vs 0.78).
   LD alone already satisfies squares on q's support; the error is on the TARGET's support
   (ld_free on Wolff samples: V_std 32 at T1.5, kite bias -1.3), which q-sampled squares never touch.
   (a) job 1046072, `loop_big.py --suite sdw` -> outputs_loop_sdw/ (45 tasks): ld + 32 sd per pair,
       w_sd in {10,100,1000}, T {1.5,2.0,2.27,2.6,3.2} x 3 seeds, iteration-matched. Ref: ld_free.
       Table: python loop_big.py summarize --suite sdw.
   (b) job 1046117, `kc_exact.py --grid kitew` -> outputs_kc_kitew/ (250 tasks, 39 symlinked refs):
       net/exact (lr 1e-3, 3e-3) and net/sampled, SK D=10 (T 1/0.4, J 0/1) + 3x3 Ising (T 2.27/1.5,
       3 seeds). Methods (kc_exact.method_weights): basis_kw{0.1,10,100} = w*kite + sd (advisor:
       upweight terminal kites), ld_basis = ld + kite + sd (advisor: full paths inform the squares),
       ld_sdw{10,100,1000}. Refs ld, basis. Table: python summarize_kc_ising3.py kitew
       (columns T=1.0/0.4 are SK, 2.27/1.5 are Ising; header says Ising for all).
   Summaries auto-written by a dependent job to figures/loop_sdk/summary_{sdw,kitew}.txt.
   RESULTS (figures/loop_sdk/summary_{sdw,kitew}.txt; advisor-facing overview: claude.ai/artifact/AiQRUBUYWAvnk2ZqkhKTeh):
   - sdw (10x10): w_sd 10/100/1000 lowers the raw sd residual up to ~4x (T2.27 7.5e-4 -> 2.0e-4; w*sd reaches
     0.2-0.7 vs ld 3-9) but KL is UNCHANGED (T2.27 2.11 vs 2.10/2.09/2.10); w1000 slightly worse at high T
     (T3.2 3.33 vs 3.18). More self-consistent != more accurate.
   - basis_kw (net/exact, geo-mean KL; Ising T2.27 | SK T0.4): kw0.1 8e-4 | 0.11; kw1 7.4e-3 | 0.25;
     kw10 1.3e-2 | 0.84; kw100 9.0e-2 | 1.92; ld 1.2e-5 | 2e-4. Upweighting terminal kites HURTS monotonically,
     downweighting helps (interior is the bottleneck), still 70-500x behind ld. Same ordering under net/sampled.
   - ld_sdw (small systems): monotonically worse with weight (exact Ising T2.27 1.2e-5 -> 2.3e-5/1.3e-4/5.7e-3).
   - ld_basis: net/exact slightly worse than ld; net/sampled 1.2-2.4x lower KL in 3/4 cells (all seeds; T0.4
     equal) BUT ~40% more wall per step, NOT compute-matched; 3b found ld_kite == ld at matched compute (tree).
     Open cheap check: ld + one-step kites (no sd), compute-matched, kc_exact + 10x10.
4e. (CANCELLED 2026-10-20, job 1045700; 47 partial results in outputs_kc_ldsd3/) LD + SMALL
   DIAMONDS vs TB vs LD, 3x3 Ising, free DAG, no replay buffer. New: `w_tb` in LoopGFlowNet (TB on either DAG, log Z group at LR_LOGZ_MULT x lr);
   kc_exact methods tb / ld_sd (+ ld_sd_dense sampled = every square per trajectory).
   `--grid ldsd3` -> outputs_kc_ldsd3/ (114 tasks: {table,net}/exact {tb,ld,ld_sd} free + {tb,ld}
   tree, net/sampled {tb,ld,ld_sd,ld_sd_dense}; T 2.27/1.5; 3 seeds). submit_kc_ldsd3.slurm.
   Table: `python summarize_kc_ising3.py ldsd3`. Loss/KL curves (SK ladder): plot_kc_ladder_loss.py.

5. Then SK N=100 low-T comparisons (no exact reference there).
   Suggested from 3b: TB_tree + ld_tree (baselines), kite_tree full band (or early, equivalent),
   optionally ld_kite_tree. Drop free-DAG configs and partial-depth kite bands.

Optional: log per-term losses (`LoopGFlowNet.last_components`) alongside the total loss.
