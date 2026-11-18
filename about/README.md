
# Environment Setup

The gflowphys project requires pytorch. All requirements are stored in the root .venv file. 

# .venv file and setup
```
gflow-phys/setup_env.sh
```

Sets up requirement installs.


## set gflow-phys environemnt (from any dir)

```bash
source gflow-phys/.venv/bin/activate
```

To check that the enviroment is working, you can check torch installation
```bash 
python -c "import torch; print(torch.__version__)"
```

## Aggregate all data
To aggregate all of the code results into a single SQL lite database, use 

```
python3 gflow-phys/aggregate_plotting/data_aggregate_to_sql.py
```

## Run local runs to generate more data 
```bash
cd gflow-phys/scripts
caffeinate -i bash run_local.sh
```


# Project Overview
This project trains neural networks to sample spin configurations according to their Boltzmann probabilities. It compares two GFlowNet training methods—Trajectory Balance (TB) and the implementation’s Kolmogorov-Criterion method (KC)—with Markov chain Monte Carlo (MCMC).

The main program runs temperature sweeps, calculates physical observables, and saves results. A separate plotting program turns those results into figures.

## Project structure

The code has been split into modules without intentionally changing the algorithms.

| Module | Contents |
|---|---|
| `physics/env.py` | Physical systems, interaction pairs, couplings, and energy calculations |
| `physics/observables.py` | Magnetization, overlap moments, susceptibility, and Binder cumulants |
| `models/policy.py` | The neural network that chooses spin values |
| `models/gflownet.py` | Shared training/sampling code, TB and KC classes, and batch-size scheduling |
| `samplers/mcmc.py` | Metropolis and Wolff samplers, plus autocorrelation diagnostics |
| `spin_glass_lib.py` | Compatibility imports and constants retained for existing scripts |
| `run_tc_sweep.py` | Command-line interface, experiment coordination, checkpointing, and data saving |
| `make_plots.py` | Plotting commands that read saved results and produce PDFs |
| `compute_tc.py` | Optional script that summarizes saved temperature estimates across seeds |
| `sk_exact_enumeration_D10.py` | Optional exact-enumeration reference for small SK systems |
| `multi_N_crossing_bootstrap.py` | Optional comparison of Binder curves between different SK sizes |

Each of `physics`, `models`, and `samplers` contains an `__init__.py` file.

Run commands from the project root so Python can find these packages.

The compatibility module imports definitions from the new modules. Those modules should not import from `spin_glass_lib.py`, because doing so can create circular imports.

Some default hyperparameter constants are temporarily duplicated between the compatibility module and the GFlowNet module. This does not affect training becuase the default hyperparams are overwritten in command line calls, but consolidating them remains a cleanup task. 

# GFlowNet Sampling for Ising and Spin-Glass Models


## Physics models

### Spins and energy

A configuration contains `D` spins. Each spin is either `-1` or `+1`.

The energy is

\[
E(s)=-\sum_{(i,j)\in\mathcal E}J_{ij}s_i s_j.
\]

There is no external magnetic field.

Positive couplings favor aligned spins; negative couplings favor opposite spins. At positive temperature `T`, the target probability is

\[
p_T(s)=\frac{\exp[-E(s)/T]}{Z(T)},
\qquad
Z(T)=\sum_s\exp[-E(s)/T].
\]

The Boltzmann constant is set to one. Lower temperature puts more probability on low-energy configurations.

The neural sampler uses the unnormalized reward

\[
R(s)=\exp[-E(s)/T].
\]

Training calculates its logarithm rather than directly exponentiating large energy values.

### Supported models

| Model | Structure | Couplings |
|---|---|---|
| `ising` | An `N × N` square grid containing `D=N²` spins | Uniform `J=+1` |
| `ea` | An `N × N` square grid containing `D=N²` spins | Independent `J=-1` with probability `p_af`, otherwise `+1` |
| `sk` | A fully connected system containing `D` spins | Independent `J=-1/√D` with probability `p_af`, otherwise `+1/√D` |

Grid models use periodic boundaries: the grid wraps around both directions.

Sites are flattened in row-major order:

```python
i = row * N + column
```

For the grids used here, with `N ≥ 3`, the interaction builder stores the right and downward bond from every site. This gives `2N²` interactions, each included once in the energy sum.

The current construction needs separate consideration for `N=1` or `N=2`, where wrapping can produce self-interactions or duplicate undirected interactions.

SK stores every pair `i < j`, giving `D(D-1)/2` interactions. Its couplings are bimodal, not Gaussian.

**Size arguments have different meanings:**

- `--N 5` means a **5×5 grid with 25 spins** for Ising or EA.
- `--D 10` means **10 total spins** for SK.

### Disorder realizations and random seeds

A disorder realization is one fixed draw of EA or SK couplings.

The runner constructs its coupling seed as

```python
J_seed = seed * 1000 + d
```

where `d` ranges from zero to `n_disorder - 1`.

For each realization:

- The couplings remain fixed across temperatures.
- TB, KC, and MCMC use the same physical environment.
- Different outer seeds normally produce different disorder sets.

The runner also seeds NumPy and PyTorch with the outer `--seed`. Therefore, for EA/SK, changing the outer seed changes both disorder and training/sampling randomness.

Ising has no random couplings. Its effective disorder count is always one; additional outer seeds provide independent training and sampling runs.

If using more than 1,000 disorder realizations per outer seed, revise the coupling-seed convention to avoid overlap between adjacent outer seeds.

## GFlowNet background

A GFlowNet learns to construct objects through a sequence of decisions so that completed objects appear in proportion to their rewards.

Here, the completed object is a spin configuration.

The policy assigns spins in a fixed order:

1. Start with every site unassigned.
2. Assign site 0.
3. Assign site 1.
4. Continue until all `D` sites are assigned.

Action `0` assigns spin `-1`; action `1` assigns spin `+1`.

Every trajectory contains exactly `D` decisions. The model does not choose the next site or learn a stopping action.

Because the assignment order is fixed, each partial configuration has one parent. Each completed configuration has one construction path. Its backward probability is therefore one, giving

```python
LOG_PB_FIXED = 0.0
```

The physical reward is evaluated after the complete configuration has been constructed.

The goal is to learn a distribution, not just find one minimum-energy configuration.

Independent neural rollouts are independent draws from the learned policy, conditional on fixed network weights. They approximate the physical distribution only to the extent that training succeeds.

## Neural network inputs and outputs

`StepEmbedPolicy` is a multilayer perceptron shared across all assignment steps.

Let:

- `B` be the batch size.
- `D` be the number of spins.
- `t` be the next site to assign.

| Quantity | Shape | Description |
|---|---|---|
| Partial spins | `(B, D)` | Assigned spins are `-1/+1`; unassigned positions are zero |
| Assignment mask | `(B, D)` | One for assigned positions, zero otherwise |
| Step argument | Python integer | The same next-site index for the entire batch |
| Learned step embedding | `(B, 16)` | Embedding for the current assignment step |
| Grid coordinates | `(B, 2)` | Ising/EA only: normalized row and column of the next site |
| Concatenated features | `(B, 2D+18)` for grids; `(B, 2D+16)` for SK | Input to the first linear layer |
| Output logits | `(B, 2)` | Scores for assigning `-1` or `+1` |
| Action probabilities | `(B, 2)` | Softmax of the logits |
| Complete action sequences | `(B, D)` | Integer actions for all trajectories |
| Complete spin configurations | `(B, D)` | Terminal `-1/+1` spins |

For a 5×5 grid:

```text
D = 25
features = 25 spins + 25 mask entries + 16 embedding values + 2 coordinates
         = 68
```

With the default batch size, each policy call maps `(16, 68)` features to `(16, 2)` logits.

**The outputs are logits, not one-hot encodings.**

The network contains three hidden linear layers of width 128, each followed by GELU, and a final linear layer producing two logits.

The step-embedding table has shape `(D, 16)`. Grid coordinates are stored in a `(D, 2)` buffer using `(row/N, column/N)`.

The input does not explicitly include the temperature or coupling matrix. Separate models are trained for each temperature and disorder realization, with optional weight transfer between temperatures.

## Shared training and sampling

### `BatchedGFlowNetBase`

This class implements the machinery shared by TB and KC:

- Construct the policy.
- Load initial policy weights when requested.
- Define exploration and reward schedules.
- Sample complete trajectories.
- Replay recorded actions with gradients.
- Run optimizer iterations.
- Generate evaluation samples.

TB and KC provide their own loss functions and optimizers.

This wrapper is not a `torch.nn.Module`. Its `train()` method performs optimization; it does not merely switch a network into training mode.

### One training iteration

Each iteration:

1. Calculates the current schedule values.
2. Samples `B` complete trajectories without gradients.
3. Replays those trajectories with gradients enabled.
4. Calculates terminal energies and the loss.
5. Clears gradients and backpropagates.
6. Clips the policy gradient norm to 1.
7. Takes an Adam optimizer step.

Sampling advances all trajectories together, one spin position at a time. Each step makes one network call for the whole batch.

Replay returns:

| Output | Shape |
|---|---|
| Trajectory log probabilities | `(B,)` |
| Mean per-step entropies | `(B,)` |
| Terminal spins | `(B, D)` |

The environment maps terminal spins to `(B,)` energies. Each method produces one scalar loss.

### Exploration and schedules

Training samples actions from

\[
q(a\mid s)=(1-\epsilon)\pi_\theta(a\mid s)+\epsilon/2.
\]

The loss evaluates the learned policy probability `πθ`, rather than the exploration mixture probability.

| Setting | Default behavior |
|---|---|
| Policy learning rate | `1e-3` |
| Training batch | 16 trajectories |
| Reward multiplier | Approximately 0.3 at the beginning, increasing to 1 |
| Exploration probability | Approximately 0.8 initially, decreasing to 0.05 |
| Entropy coefficient | Approximately 0.05 initially, decreasing to zero |
| Entropy annealing duration | First 80% of training |
| Evaluation exploration | Zero |

The implementation calls the reward multiplier `beta`, but physical inverse temperature is `1/T`.

Training uses

```python
logR = beta * (-energy / T)
```

so it gradually changes the training target toward the requested Boltzmann distribution. It does not multiply policy logits by `beta`.

Schedules use `iteration / n_iter`, starting at iteration one. Consequently, initial values are slightly beyond their nominal starting values.

Increasing `n_iter` changes the schedule as well as the update count. A 3,000-iteration run is not simply an identical 1,500-iteration run followed by extra updates.

### Trajectory Balance

`TBGFlowNet` learns the policy and a scalar `log_Z`.

Its residual is

\[
r=\log Z_\theta+\log P_F(\tau)-\log R(s),
\]

and its loss is

\[
L_{TB}=\operatorname{mean}(r^2)
-\lambda_H\operatorname{mean}(H).
\]

The backward-log-probability term is zero for this fixed-order construction.

`log_Z` starts at `D log(2)` unless overridden. It uses a separate Adam parameter group with learning rate

```python
LR_NN * lr_logz_mult
```

The default multiplier is 50, giving a normalizer learning rate of 0.05.

Gradient clipping applies to policy parameters; it does not include the separate `log_Z` parameter.

### KC

`KCGFlowNet` avoids fitting `log_Z`. It calculates

\[
\delta=\log P_F(\tau)-\log R(s)
\]

and minimizes

\[
L_{KC}=\operatorname{Var}(\delta)
-\lambda_H\operatorname{mean}(H).
\]

The implementation uses population variance:

```python
residual.var(unbiased=False)
```

At the target distribution, the residual is the same for every trajectory: `-log Z`. Some original comments incorrectly give that constant a positive sign.

The method computes variance directly; it does not allocate all trajectory pairs.

**Use a training batch larger than one.** With one trajectory, the variance is identically zero and supplies no balance-loss learning signal.

TB and KC loss magnitudes should not be compared directly as a measure of sampling quality.

## MCMC sampling

`run_mcmc` builds a fresh chain at each temperature.

With `--mcmc_method auto`:

- Ising uses Wolff cluster updates.
- EA and SK use single-spin Metropolis updates.

After warm-up, the chain performs `mcmc_thin` updates between retained configurations.

### Metropolis

Each update proposes flipping one randomly chosen spin.

It calculates the local energy change

\[
\Delta E=2s_i\sum_jJ_{ij}s_j.
\]

The flip is accepted automatically when `ΔE ≤ 0`, otherwise with probability `exp(-ΔE/T)`.

A Metropolis update is **one proposed spin flip**, not a complete lattice sweep.

### Wolff

Each update grows and flips a cluster of aligned spins.

The addition probability for uniform `J=1` is

\[
p_{\mathrm{add}}=1-\exp(-2/T).
\]

This implementation is valid only for the ferromagnetic Ising model. Forcing Wolff on EA or SK is rejected.

One Wolff update is a whole cluster flip. Its warm-up and thinning counts therefore represent different work from the same counts under Metropolis.

### Diagnostics

| Diagnostic | Meaning |
|---|---|
| `acceptance_rate` | Accepted fraction of Metropolis proposals, including warm-up counters |
| `mean_cluster_fraction` | Mean Wolff cluster size divided by `D`, including recorded warm-up clusters |
| `tau_int` | Estimated integrated autocorrelation time of the post-warm-up energy trace |

`NaN` acceptance for Wolff is expected. Cluster fraction is correspondingly `NaN` for Metropolis.

When diagnostics are enabled, energy is recorded after every raw post-warm-up update. The trace length is

```python
n_eval * mcmc_thin
```

`integrated_autocorr_time` uses an FFT and an automatic window. Its result measures energy correlation in raw update units. It does not prove equilibration or good mixing of every observable.

Thinning and random pairing do not repair a chain that has not equilibrated.

## Physical observables

### Ising susceptibility

For each configuration,

\[
m=\frac{1}{D}\sum_i s_i.
\]

The code calculates

\[
\chi=\frac{D}{T}
\left(\langle m^2\rangle-\langle|m|\rangle^2\right).
\]

This is an absolute-magnetization finite-size convention. It differs from a formula using `⟨m⟩²`.

Use the same definition when comparing with another implementation.

### Spin-glass overlap

Two configurations from the same disorder realization define

\[
q_{ab}=\frac{1}{D}\sum_i s_i^{(a)}s_i^{(b)}.
\]

The code randomly permutes configurations and forms disjoint pairs. With 800 configurations, it normally calculates 400 overlaps.

It then computes

\[
\chi_{SG}=D\langle q^2\rangle
\]

and

\[
U_4=1-\frac{\langle q^4\rangle}
{3\langle q^2\rangle^2+10^{-12}}.
\]

The implemented spin-glass susceptibility has no extra temperature factor.

Susceptibility and Binder functions generate their pairings separately, so their finite-sample moments need not come from identical pairs.

### Temperature estimates

`peak_estimate` finds the largest susceptibility and uses neighboring points for an interpolated peak position when possible.

`crossover_from_binder` finds the interval where one Binder curve has its largest absolute slope.

**The saved `Tcross_*_binder` value is a steepest-slope crossover, not an intersection of curves from two sizes.**

A finite 5×5 system has rounded behavior. These estimates are finite-system diagnostics, not automatically thermodynamic critical temperatures.

## Function reference

| Definition | What it does |
|---|---|
| `build_neighbour_list(N)` | Returns wrapped grid interaction pairs |
| `build_all_pairs(D)` | Returns all fully connected pairs with `i < j` |
| `SpinEnv` | Stores the system, couplings, neighbor lists, and cached interaction arrays |
| `energy_batch(X)` | Maps Torch configurations `(B,D)` to energies `(B,)` |
| `energy_fast_np(x)` | Maps one NumPy configuration `(D,)` to a scalar energy |
| `local_delta_energy(x,i)` | Calculates the energy change for flipping site `i` |
| `StepEmbedPolicy.forward` | Maps partial configurations, masks, and a step index to `(B,2)` logits |
| `_sample_actions_batch` | Samples integer action sequences `(B,D)` without gradients |
| `_replay_batch` | Recalculates log probabilities, entropy, and final spins for fixed actions |
| `train` | Runs optimizer iterations; returns whether training completed |
| `sample(n)` | Returns a list of `n` NumPy spin vectors, each `(D,)` |
| `state_dict` | Returns policy weights only; excludes optimizer and TB normalizer |
| `make_batch_size_schedule` | Returns the callable used to interpolate training batch sizes |
| `run_chain` | Runs warm-up and retained MCMC sampling; optionally returns an energy trace |
| `run_mcmc` | Chooses a sampler and optionally returns diagnostics |
| `integrated_autocorr_time` | Estimates autocorrelation time for a one-dimensional trace |
| `magnetization_susceptibility` | Returns Ising susceptibility and mean absolute magnetization |
| `spin_glass_overlap_moments` | Returns overlap samples and their second/fourth moments |
| `spin_glass_susceptibility` | Returns spin-glass susceptibility and mean absolute overlap |
| `binder_cumulant_sg` | Returns the overlap Binder cumulant |
| `susceptibility` | Chooses the appropriate susceptibility calculation |
| `order_parameter_samples_for_hist` | Returns magnetizations or overlaps for histogramming |
| `run_sweep_for_disorder` | Runs temperatures for one fixed coupling realization |
| `save_checkpoint` / `load_checkpoint` | Save and restore completed-temperature progress |
| `load_seed_jsons` | Loads matching result files from `seed_*` subdirectories |
| `stack_and_average` | Calculates pointwise means and SEM across outer seeds |

Legacy tuple-based environment methods remain available, including `root`, `step`, `encode`, `energy`, and `log_reward`. The current batched training path does not use them.

## Installation

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib
```

Check the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('Apple GPU available:', torch.backends.mps.is_available())"
```

Record a working environment:

```bash
python -m pip freeze > requirements-lock.txt
```

Activate the environment again in each new terminal:

```bash
source .venv/bin/activate
```

## Command-line usage

The general command is:

```bash
python run_tc_sweep.py --model MODEL --T_min LOW --T_max HIGH [OPTIONS]
```

The required arguments are `--model`, `--T_min`, and `--T_max`.

### Model and temperature arguments

| Argument | Default | Meaning |
|---|---|---|
| `--model` | Required | `ising`, `ea`, or `sk` |
| `--N` | `5` | Grid side length for Ising/EA; total spins are `N²` |
| `--D` | `5` | Total spins for SK |
| `--p_af` | `0.5` | Probability of a negative EA/SK coupling |
| `--T_min` | Required | Lowest temperature; use a positive value |
| `--T_max` | Required | Highest temperature |
| `--n_T` | `14` | Number of evenly spaced temperature points |

`--p_af` does not change Ising couplings. Changing it away from 0.5 changes the mean SK coupling, so the physical interpretation also changes.

Use at least two temperatures for the slope calculation; a meaningful temperature study generally needs substantially more.

### Training arguments

| Argument | Default | Meaning |
|---|---|---|
| `--methods` | `TB,KC,MCMC` | Comma-separated methods to run |
| `--n_iter` | `1500` | Training updates per neural method, temperature, and disorder realization |
| `--batch_start` | `16` | Initial training batch-size schedule endpoint |
| `--batch_end` | `16` | Final training batch-size schedule endpoint |
| `--entropy_coef0` | `0.05` | Initial entropy regularization coefficient |
| `--entropy_anneal_frac` | `0.8` | Fraction of training over which entropy regularization decreases to zero |
| `--lr_logz_mult` | `50` | TB normalizer learning-rate multiplier |
| `--init_log_z_override` | None | Explicit initial TB log normalizer |
| `--warm_start` | Off | Transfer policy weights and TB normalizer between temperatures |
| `--device` | `cpu` | Device for neural computation |

The policy learning rate, hidden widths, embedding dimension, and exploration/reward schedule endpoints are currently module constants rather than CLI arguments.

With warm start disabled, temperatures run low to high with newly initialized models.

With warm start enabled, temperatures run high to low. Policy weights and TB `log_Z` transfer, but optimizer state does not. Training schedules restart at each temperature.

An explicit `--init_log_z_override` takes precedence over the transferred normalizer.

### Sampling and randomness arguments

| Argument | Default | Meaning |
|---|---|---|
| `--n_eval` | `800` | Retained configurations per method and temperature |
| `--n_disorder` | `1` | EA/SK coupling realizations per outer seed |
| `--seed` | `0` | Outer training/sampling seed and coupling-seed prefix |
| `--mcmc_warmup` | `50000` | Raw MCMC updates before retention |
| `--mcmc_thin` | `50` | Raw updates between retained configurations |
| `--mcmc_method` | `auto` | `auto`, `metropolis`, or `wolff` |
| `--mcmc_diagnostics` | On | Calculate energy-based MCMC diagnostics |
| `--no_mcmc_diagnostics` | Not selected | Disable those diagnostics |

`n_eval` is independent of the neural training batch size.

Increasing `n_eval` improves sampling precision but does not necessarily fix training bias or MCMC equilibration.

Ising always uses one effective disorder realization.

### Saving and logging arguments

| Argument | Default | Meaning |
|---|---|---|
| `--outdir` | `./outputs` | Directory for this run’s files |
| `--no_resume` | Off | Ignore an existing checkpoint and start fresh |
| `--log_loss` | Off | Record selected neural training-loss histories |
| `--log_loss_every` | `25` | Loss-recording interval |
| `--log_loss_T_indices` | `0,-1` | Temperature-grid indices to record |
| `--disagree_tol` | `0.35` | Relative susceptibility difference from MCMC to flag |

Negative temperature indices count from the end. Loss logging also records the first and final iteration.

The disagreement threshold is a warning criterion, not a statistical significance test.

### Quick smoke test

This checks imports, training, sampling, and writing outputs. Its results are not converged physics estimates.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 2 --n_iter 10 --n_eval 50 --n_disorder 1 --mcmc_warmup 100 --mcmc_thin 1 --device cpu --no_resume --outdir ./smoke_test_refactor
```

`--no_resume` ensures repeated tests execute again.

### Short timing test

This command took **6.735 seconds** on the measured M1 Pro setup:

```bash
time OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 2 --n_iter 100 --n_eval 200 --n_disorder 1 --mcmc_warmup 1000 --mcmc_thin 10 --device cpu --outdir ./timing_cpu_100
```

### Full 5×5 temperature sweep

This command took **8 minutes 55.62 seconds** after refactoring:

```bash
time OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 14 --n_iter 1500 --n_eval 800 --n_disorder 1 --seed 0 --mcmc_warmup 50000 --mcmc_thin 50 --device cpu --log_loss --outdir ./outputs/ising_5x5_refactored/seed_0
```

### Longer training comparison

This preserves the same physics and evaluation settings but doubles neural training iterations:

```bash
time OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 14 --n_iter 3000 --n_eval 800 --n_disorder 1 --seed 0 --mcmc_warmup 50000 --mcmc_thin 50 --device cpu --log_loss --outdir ./outputs/ising_5x5_iter3000/seed_0
```

Expected time is approximately **15–20 minutes**, extrapolated rather than measured.

Use a separate base directory so the plotting program does not average different training settings together.

### Multiple independent runs

Run sequential seeds with distinct output directories:

```bash
for seed in {0..9}; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_tc_sweep.py \
    --model ising --N 5 --methods TB,KC,MCMC \
    --T_min 2 --T_max 3 --n_T 14 \
    --n_iter 1500 --n_eval 800 --n_disorder 1 \
    --seed "$seed" --mcmc_warmup 50000 --mcmc_thin 50 \
    --device cpu --log_loss \
    --outdir "./outputs/ising_5x5_refactored/seed_${seed}" \
    > "ising_5x5_seed_${seed}.log" 2>&1 || break
done
```

Ten comparable seeds would take roughly **89 minutes sequentially** based on the measured full-run timing.

Select only unfinished seeds if some already completed. Completed runs delete their checkpoints; repeating their commands can retrain and overwrite results.

## Output data

### Where results are stored

Outputs go to the directory specified by `--outdir`, relative to the directory where the command is launched unless an absolute path is supplied.

Recommended layout:

```text
outputs/
  ising_5x5_refactored/
    seed_0/
    seed_1/
  ising_5x5_iter3000/
    seed_0/
```

The plotter expects per-seed folders named `seed_*`.

File tags are:

| Model | Tag |
|---|---|
| Ising | `ising` |
| EA | `ea` |
| SK with 10 spins | `sk_D10` |

Each run uses fixed filenames within its output directory. Never send concurrent jobs to the same directory.

### Saved files

| File | Contents |
|---|---|
| `{tag}_chi_sweep.json` | Aggregate curves, uncertainty arrays, temperature estimates, and diagnostic arrays |
| `{tag}_raw_data.npz` | Endpoint spin configurations, selected loss histories, and temperatures |
| `{tag}_tc_estimates.txt` | Human-readable settings and summaries |
| `{tag}_checkpoint.pt` | Intermediate progress; removed after successful completion |

No plots are generated by the sweep itself.

### JSON contents

The JSON contains:

- Model name, `N`, `D`, method list, temperature grid, and effective disorder count.
- `chi_tb_mean`, `chi_kc_mean`, and `chi_mcmc_mean`, as applicable.
- Corresponding `_sem` arrays.
- Susceptibility peak estimates such as `Tc_tb_chi_peak`.
- EA/SK Binder means, SEMs, and steepest-slope estimates.
- MCMC diagnostic means and SEMs.

Curve arrays have length `n_T`.

Within one run, means and SEMs are calculated across disorder realizations. With one realization, those SEM arrays are zero; this does not mean the observables have no sampling uncertainty.

The JSON is not a complete record of every setting. Preserve the command, logs, environment versions, and code revision.

**Grid metadata caveat:** the saved JSON `D` field comes from the runner’s `--D` argument even for grid models. Use `N²` for the actual Ising/EA spin count.

Diagnostic `NaN` values may appear in the JSON. Strict JSON readers may need to handle those explicitly.

### Raw NPZ contents

Example keys include:

```text
T_grid
samples_lowT_tb
samples_lowT_tb_T
samples_highT_tb
samples_highT_tb_T
samples_lowT_kc
samples_highT_kc
samples_lowT_mcmc
samples_highT_mcmc
loss_tb_Tidx0
loss_tb_Tidx0_T
```

Sample arrays have:

```text
shape: (n_eval, D)
dtype: float32
values: -1 or +1
```

Loss arrays have:

```text
shape: (n_records, 2)
dtype: float64
columns: iteration, loss
```

**Retention limits:**

- Final raw samples are saved only for the lowest and highest temperatures.
- They come only from disorder realization zero.
- Loss histories also come only from realization zero, at requested temperature indices.
- Intermediate-temperature samples are not retained in the final NPZ.
- The completed run does not export every trained model for later reuse.

The runner temporarily retains more first-disorder samples during execution and checkpointing than it ultimately writes to the final NPZ.

### Approximate file sizes

For the measured configuration:

```text
25 spins
800 samples
3 methods
2 endpoint temperatures
float32 values
```

The raw spin arrays alone require

```text
25 × 800 × 3 × 2 × 4 bytes = 480,000 bytes
```

That is approximately **0.48 MB**, before archive headers, temperatures, and loss records.

With 1,500 iterations, loss logging every 25 iterations, two logged temperatures, and two neural methods, the loss arrays add approximately **3.9 KB**.

| Artifact | Approximate scale for this 5×5 setup |
|---|---|
| Raw NPZ | Around half a megabyte |
| JSON | Usually kilobytes to tens of kilobytes |
| Text summary | Usually a few kilobytes |
| Checkpoint | Variable; can exceed final data because it retains intermediate samples and state |
| Plot PDFs | Variable with figure content |

These are estimates except for the calculated array payloads. Inspect actual sizes with:

```bash
du -h ./outputs/ising_5x5_refactored/seed_0/*
```

Final raw-sample storage grows approximately with `D × n_eval × number_of_methods`. Increasing `n_T` grows aggregate curves and may grow loss logs, but final raw configurations still contain only two temperatures.

### What makes two runs different?

| Changed setting | What changes |
|---|---|
| `seed` | Training/sampling randomness; also EA/SK couplings |
| `N` or `D` | Physical system size and network input dimensions |
| `p_af` | EA/SK coupling distribution |
| Temperature range/grid | Which physical distributions are sampled |
| `n_iter` | Training budget and schedule progression |
| `batch_start/end` | Number of trajectories per update and stochastic loss behavior |
| `n_eval` | Number of evaluation configurations and estimator precision |
| MCMC warm-up/thinning | Chain work and retained-sample correlation |
| `n_disorder` | Number of coupling realizations averaged within a run |
| `warm_start` | Temperature order and transfer of learned weights |
| `methods` | Which algorithms execute; also changes subsequent random-number consumption |
| Device/backend | Performance and potentially numerical/reproducibility behavior |

Keep different experimental settings in different base directories. Seeds intended for averaging should otherwise use matching settings.

## Plotting

Create aggregate curves:

```bash
python make_plots.py chi --base_dir ./outputs/ising_5x5_refactored --tag ising --model_kind ising --outdir ./figures/ising_5x5_refactored
```

Create endpoint histograms:

```bash
python make_plots.py distributions --seed_dir ./outputs/ising_5x5_refactored/seed_0 --tag ising --model_kind ising --outdir ./figures/ising_5x5_refactored
```

Create training-loss plots:

```bash
python make_plots.py loss --seed_dir ./outputs/ising_5x5_refactored/seed_0 --tag ising --outdir ./figures/ising_5x5_refactored
```

Create MCMC diagnostics:

```bash
python make_plots.py mcmc_diag --base_dir ./outputs/ising_5x5_refactored --tag ising --outdir ./figures/ising_5x5_refactored
```

On macOS:

```bash
open ./figures/ising_5x5_refactored
```

The plotter writes PDFs using a noninteractive backend.

`chi` averages saved mean curves across outer seeds and calculates SEM across those seeds. It does not propagate each run’s internal disorder SEM. With one outer seed, its SEM is zero.

The loader searches:

```text
base_dir/seed_*/{tag}_chi_sweep.json
```

It does not read a JSON directly inside `base_dir`. The distribution command can read a run directory directly through `--seed_dir`.

## Checkpointing and restarting

Progress is saved after a whole temperature point completes all requested methods.

Restart using the same command and output directory without `--no_resume`.

Important limitations:

- Interrupted neural training at the current temperature is discarded.
- Completed temperatures can be reused.
- Optimizer state is not saved.
- Random-number-generator states are not saved.
- Resumed runs are not guaranteed to match uninterrupted runs bit for bit.
- Changed argument signatures cause a warning and fresh execution.
- Source-code changes are not comprehensively detected.
- Final successful execution removes the checkpoint.

The runner handles `SIGTERM` and `SIGUSR1`. During neural training, it checks for interruption between updates. During MCMC, it may need to finish the current chain before responding.

A gracefully interrupted run exits with status zero, so exit status alone does not prove completion.

Only load trusted checkpoints: the current loader allows Python object deserialization.

## Parallelization and optimization

### Independent experiments

The current runner executes sequentially.

The simplest parallel work units are independent outer seeds. Give each worker:

- Its own seed.
- Its own output directory.
- A controlled CPU thread count.

Disorder realizations currently loop inside the runner. Parallelizing them requires additional coordination and aggregation.

Warm-started temperatures depend on previous weights and must remain ordered. Splitting non-warm-start temperatures is possible in principle but requires output merging and changes random-number consumption.

### CPU threading

The measured baseline uses:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Compiled library kernels may use threads, but these settings do not parallelize Python loops.

Small network operations can become slower with excessive threading. Benchmark one, two, and four threads rather than assuming more is better.

With multiple processes, keep total thread usage within available or allocated CPU resources.

### GPU execution

Test `--device mps` on Apple hardware or `--device cuda` on compatible NVIDIA hardware.

This moves neural computations and Torch energy calculations. MCMC remains CPU code.

The small network, batch size 16, and sequential assignment decisions may make GPU overhead dominate. Benchmark before paying for GPU access.

### Candidate improvements

These were suggested but are not part of the structural refactor:

1. Convert sampled actions directly to terminal spins instead of replaying the network during evaluation.
2. Batch fixed-action training replay across timesteps.
3. Compile array-based MCMC update loops.
4. Benchmark thread counts and larger batches.
5. Reduce diagnostic overhead only after establishing trustworthy sampling settings.

Replay batching requires an interface accepting a different step index per row. Sampling still depends on previous decisions.

## Running on a cluster

A cluster mainly helps run many seeds simultaneously. A single cluster CPU may not outperform the Mac.

Create a fresh Linux environment following the institution’s instructions. Do not copy the Mac virtual environment onto the cluster.

The following Slurm example runs ten seeds with at most four concurrent tasks. Adjust account, partition, Python setup, memory, and wall time for the institution.

Save as `run_ising_array.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=ising5
#SBATCH --array=0-9%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/ising_%A_%a.out
#SBATCH --error=logs/ising_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load the institution's Python module here if required.
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

seed="$SLURM_ARRAY_TASK_ID"

srun python -u run_tc_sweep.py \
  --model ising --N 5 --methods TB,KC,MCMC \
  --T_min 2 --T_max 3 --n_T 14 \
  --n_iter 1500 --n_eval 800 --n_disorder 1 \
  --seed "$seed" \
  --mcmc_warmup 50000 --mcmc_thin 50 \
  --device cpu --log_loss \
  --outdir "./outputs/ising_5x5_cluster/seed_${seed}"
```

Submit from the project root:

```bash
mkdir -p logs
sbatch run_ising_array.sbatch
squeue -u "$USER"
```

Create `logs` before submission because Slurm opens log files before executing the script.

The early signal gives the program an opportunity to checkpoint. Confirm signal delivery and restart behavior on the actual cluster. Checkpointing does not automatically resubmit the job.

Memory and time requests are starting allocations, not measured requirements.

If cluster jobs match the Mac’s nine-minute timing, ten seeds on four simultaneous workers would need roughly three waves, or 27 minutes of running time, plus queue delay and overhead.

Do not run production sweeps on login nodes. Install dependencies once using the institution’s approved workflow.

## Exact enumeration

The supplied exact script enumerates all `2ᴰ` configurations, then explicitly constructs every state-pair overlap.

At `D=10`, this is practical:

```bash
python sk_exact_enumeration_D10.py --D 10 --p_af 0.5 --T_min 0.2 --T_max 0.4 --n_T 2 --seed 0 --n_disorder 3 --outdir ./exact_D10
```

Match temperatures, coupling probability, and seeds when comparing with sampled SK results.

The pairwise implementation scales as `O(4ᴰ)`. One float64 overlap matrix requires:

| Spin count | Matrix storage |
|---|---|
| 10 | 8 MiB |
| 14 | 2 GiB |
| 25 | 8 PiB |

The code allocates several such matrices and temporaries. Blocking reduces peak memory but does not remove the total pairwise work.

A proposed Walsh–Hadamard/XOR-convolution replacement could reduce the overlap calculation to `O(D2ᴰ)` work and `O(2ᴰ)` memory. It has not been implemented or timed.

The exact helper’s returned `Z` is also shifted by its numerical-stability transformation. For the physical normalizer, preserve the original maximum log weight and calculate:

```python
logZ = original_logw_max + np.log(w.sum())
```

Normalized probabilities and overlap moments are unaffected by that shift.

## Validation and interpretation

A successful run proves that the code executes and saves results. It does not prove that either neural sampler or MCMC has converged.

Before drawing conclusions:

- Compare multiple outer seeds.
- Check sensitivity to training length and evaluation count.
- Inspect endpoint distributions and training histories.
- Check MCMC equilibration and observable-specific mixing.
- Validate small SK systems against exact enumeration.
- Keep observable definitions consistent.
- Separate finite-size crossovers from thermodynamic transition claims.

The optional `compute_tc.py` averages saved per-seed temperature estimates. That differs from locating a peak on a seed-averaged curve.

The optional bootstrap script studies intersections between different SK sizes. Its historical `--N_values` refers to SK spin counts, with directories such as `D10` and `D25`, rather than grid side lengths.


# Plotting Sergio's saved results from SQLite

This guide matches the **short import scripts you ran in this conversation**. It does not require the separate `sergio_results.py` package.

Your database is:

```text
path/to/gflow-phys/aggregate_plotting/sergio_results.sqlite
```

Run the Python examples in order in a notebook, Python interactive session, or a script saved in `aggregate_plotting`. Later examples reuse the helpers from earlier sections. Plots are saved in `aggregate_plotting/figures`.

## 1. Activate the environment

From the project root if haven't already:

```bash
source .venv/bin/activate
```


## 2. Understand the actual database layout

| Table | Contents |
|---|---|
| `archived_files` | Original JSON, TXT, and NPZ bytes; columns `path`, `extension`, `content` |
| `npz_catalog` | Maps each original NPZ path and array key to an unpacked SQL table; also stores original shape and dtype |
| `npz_<hash>_<width>` | Unpacked array rows; identifiers `source_path`, `array_key`, `row_index`, followed by numeric columns `value_0`, `value_1`, etc. |

The hash in an NPZ table name comes from the array key. **Use `npz_catalog` to locate arrays; do not guess table names.** Different files with the same array key and width share a table, distinguished by `source_path` and `array_key`.

For a `(800, 25)` sample array, each SQL row is one configuration and the 25 value columns are spins. For a `(n_records, 2)` loss array, `value_0` is iteration and `value_1` is loss. A scalar temperature is stored as one row with `value_0`.

The JSON records have been archived, not converted into a normalized `runs` or `measurements` table. The examples below decode them into DataFrames. TXT summaries are still text: they have not been converted into plot-ready summary measurements.

## 3. Connect without modifying the database

```python
from pathlib import Path
import io
import json
import re
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DB = Path('path/to/gflow-phys/aggregate_plotting/sergio_results.sqlite')
if not DB.is_file():
    raise FileNotFoundError(DB)
conn = sqlite3.connect(DB.resolve().as_uri() + '?mode=ro', uri=True)
conn.execute('PRAGMA query_only=ON')
FIGURES = DB.parent / 'figures'
FIGURES.mkdir(exist_ok=True)

print(pd.read_sql_query(
    'SELECT extension, COUNT(*) AS files FROM archived_files GROUP BY extension', conn))
print(pd.read_sql_query(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn))
```

Read-only mode prevents an accidental update and avoids silently creating an empty database if the path is wrong.

## 4. Inspect every available NPZ array

```python
catalog = pd.read_sql_query(
    'SELECT source_path,array_key,table_name,shape,dtype FROM npz_catalog ORDER BY source_path,array_key',
    conn)
print(catalog.to_string(index=False))
print('\nSample arrays:')
print(catalog[catalog.array_key.str.startswith('samples_')].to_string(index=False))
print('\nLoss arrays:')
print(catalog[catalog.array_key.str.startswith('loss_')].to_string(index=False))
```

Typical keys are `samples_lowT_tb`, `samples_highT_kc`, their `_T` scalar companions, and `loss_tb_Tidx0` with `loss_tb_Tidx0_T`. Use the catalog's actual keys; a run without loss logging will not have loss arrays.

## 5. Load a selected NPZ array as a DataFrame

```python
def array_rows(source_path, array_key, limit=None):
    entry = pd.read_sql_query(
        'SELECT table_name,shape,dtype FROM npz_catalog WHERE source_path=? AND array_key=?',
        conn, params=[str(source_path), array_key])
    if len(entry) != 1:
        raise KeyError(f'Expected one array: {source_path} / {array_key}')
    table = entry.iloc[0].table_name
    if not re.fullmatch(r'npz_[0-9a-f]{16}_\d+', table):
        raise ValueError(f'Unexpected SQL table name: {table}')
    sql = f'SELECT * FROM "{table}" WHERE source_path=? AND array_key=? ORDER BY row_index'
    params = [str(source_path), array_key]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        sql += ' LIMIT ?'
        params.append(int(limit))
    frame = pd.read_sql_query(sql, conn, params=params)
    expected = json.loads(entry.iloc[0]['shape'])
    expected_rows = expected[0] if expected else 1
    if limit is None and len(frame) != expected_rows:
        raise ValueError('Incomplete array import; rerun the NPZ import before plotting')
    return frame


def value_columns(frame):
    return sorted((c for c in frame if c.startswith('value_')),
                  key=lambda c: int(c.split('_')[1]))


def array_temperature(source_path, array_key):
    scalar = array_rows(source_path, array_key + '_T')
    return float(scalar['value_0'].iloc[0])
```

Identifiers are validated separately; source paths and keys use bound SQL parameters. `limit=5` is useful for previewing rows, but never use a preview to calculate a full-sample observable.

## 6. Choose one experiment and avoid copied files

```python
EXPERIMENT_FRAGMENT = '/GFN_Thesis/outputs/ising_5x5_refactored/'
selected_catalog = catalog[
    catalog.source_path.str.contains(EXPERIMENT_FRAGMENT, regex=False)
    & ~catalog.source_path.str.contains('/plot_inputs/', regex=False)
].copy()
if selected_catalog.empty:
    raise ValueError('No matching experiment; inspect catalog and edit EXPERIMENT_FRAGMENT')

sample_sources = selected_catalog.loc[
    selected_catalog.array_key.eq('samples_lowT_tb'), 'source_path'].unique()
if len(sample_sources) == 0:
    raise ValueError('This experiment has no samples_lowT_tb array')
SOURCE = sorted(sample_sources)[0]
print('Selected NPZ:', SOURCE)
print(array_rows(SOURCE, 'samples_lowT_tb', limit=5))
```

Keep research sweeps, smoke tests, and timing tests separate. The short import scripts retain copied `plot_inputs` files as additional sources; they do not automatically deduplicate experiments. Filtering to a single canonical experiment prevents counting a copy twice. Distinct seeds remain distinct runs.

## 7. Plot endpoint magnetization distributions by method

```python
fig, ax = plt.subplots(figsize=(7, 4))
for method in ['tb', 'kc', 'mcmc']:
    key = f'samples_lowT_{method}'
    if not ((selected_catalog.source_path == SOURCE) & (selected_catalog.array_key == key)).any():
        continue
    frame = array_rows(SOURCE, key)
    X = frame[value_columns(frame)].to_numpy(dtype=float)
    magnetization = X.mean(axis=1)
    T = array_temperature(SOURCE, key)
    ax.hist(magnetization, bins=np.linspace(-1.05, 1.05, 40), density=True,
            histtype='step', linewidth=1.8, label=f'{method.upper()} (T={T:g})')
ax.set(xlabel='Magnetization m', ylabel='Probability density',
       title='Low-temperature spin configurations: one seed')
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / 'lowT_magnetization.pdf')
plt.show()
plt.close(fig)
```

Change `lowT` to `highT` in both the selection and plotting keys for high-temperature samples. These are terminal spin configurations, not the sequence of partially assigned states. Endpoint samples are from disorder realization zero in the legacy sweep exports.

## 8. Plot saved training loss versus iteration

```python
loss_entries = selected_catalog[
    selected_catalog.source_path.eq(SOURCE)
    & selected_catalog.array_key.str.match(r'^loss_.+_Tidx-?\d+$')
]
fig, ax = plt.subplots(figsize=(7, 4))
for entry in loss_entries.itertuples(index=False):
    frame = array_rows(entry.source_path, entry.array_key)
    T = array_temperature(entry.source_path, entry.array_key)
    ax.plot(frame['value_0'], frame['value_1'], label=f'{entry.array_key}, T={T:g}')
ax.set(xlabel='Training iteration', ylabel='Training loss', title='Saved training histories')
if not loss_entries.empty:
    ax.legend(fontsize=8)
else:
    print('No loss histories found for this source; logging may have been disabled.')
fig.tight_layout()
fig.savefig(FIGURES / 'training_loss.pdf')
plt.show()
plt.close(fig)
```

Plot comparable objectives together when interpreting loss values. TB and KC losses have different meanings; a lower numerical loss does not establish a more accurate sampler. For logarithmic y axes, first confirm plotted losses are strictly positive: entropy-regularized total losses can be zero or negative.

## 9. Turn legacy JSON sweeps into a plotting DataFrame

This loader recognizes `T_grid`, `methods`, and keys such as `chi_tb_mean` and `u4_kc_mean`. It skips JSON diagnostic reports with a different structure. It reads JSON files sequentially and keeps only the small scalar plotting records in memory.

```python
records = []
for path, content in conn.execute(
        "SELECT path,content FROM archived_files WHERE extension='.json' ORDER BY path"):
    if EXPERIMENT_FRAGMENT not in path or '/plot_inputs/' in path:
        continue
    payload = json.loads(content)
    if not isinstance(payload, dict) or 'T_grid' not in payload or 'methods' not in payload:
        continue
    found_seed = re.search(r'/seed_(\d+)(?:/|$)', path)
    seed = payload.get('seed', int(found_seed.group(1)) if found_seed else None)
    temperatures = payload['T_grid']
    for method in payload['methods']:
        method = method.lower()
        for metric in ['chi', 'u4']:
            key = f'{metric}_{method}_mean'
            if key not in payload:
                continue
            values = payload[key]
            errors = payload.get(f'{metric}_{method}_sem', [None] * len(temperatures))
            if len(values) != len(temperatures) or len(errors) != len(temperatures):
                raise ValueError(f'Inconsistent temperature grid: {path}, {key}')
            for T, value, reported_sem in zip(temperatures, values, errors):
                records.append(dict(source_path=path, seed=seed, method=method,
                    metric=metric, T=float(T), value=value, reported_sem=reported_sem,
                    n_disorder=payload.get('n_disorder')))
sweeps = pd.DataFrame(records)
if sweeps.empty:
    raise ValueError('No supported sweep JSONs matched the selected experiment')
print(sweeps.head())
```

For Ising, `chi` is the magnetization susceptibility. For EA/SK, the legacy field named `chi` represents spin-glass susceptibility. Do not combine these definitions in one estimate. For legacy grid files, actual spin count is `N**2`; the saved JSON `D` field can reflect an unused SK CLI default.

## 10. Reproduce susceptibility versus temperature with seed SEM

```python
chi = sweeps[sweeps.metric.eq('chi')].dropna(subset=['value']).copy()
if chi.seed.isna().any():
    raise ValueError('Seed metadata is required for across-seed uncertainty')
if chi.duplicated(['method', 'T', 'seed']).any():
    raise ValueError('Multiple rows per method/T/seed: check experiment filters and copies')
stats = chi.groupby(['method', 'T'])['value'].agg(['mean', 'std', 'count']).reset_index()
stats['sem'] = stats['std'] / np.sqrt(stats['count'])
fig, ax = plt.subplots(figsize=(7, 4))
for method, group in stats.groupby('method'):
    group = group.sort_values('T')
    ax.plot(group['T'], group['mean'], marker='o', label=method.upper())
    valid = group['count'] > 1
    ax.errorbar(group.loc[valid, 'T'], group.loc[valid, 'mean'],
                yerr=group.loc[valid, 'sem'], fmt='none', capsize=3)
ax.set(xlabel='Temperature T', ylabel='Ising susceptibility χ',
       title='5×5 Ising: mean and SEM across independent outer seeds')
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / 'ising_chi_across_seeds.pdf')
plt.show()
plt.close(fig)
```

This reproduces the averaging convention of `make_plots.py chi`: it averages each seed's saved curve and computes SEM across those seeds. It does not propagate each run's internal disorder SEM. A single seed has no measurable across-seed SEM, so its error bar is omitted. For an SK/EA Binder plot, select `metric='u4'` and change the y label. Ising sweep files may have no `u4` series.

## 11. Cross-plot a sample-derived observable against a saved JSON metric

This example relates each seed/method's mean absolute magnetization from its **low-temperature NPZ samples** to susceptibility from its JSON file at that same temperature. There is one plotted point per seed/method. It does not pool independent experiment settings or average the points first.

```python
endpoint_records = []
entries = selected_catalog[
    selected_catalog.array_key.str.match(r'^samples_lowT_(tb|kc|mcmc)$')
]
for entry in entries.itertuples(index=False):
    method = entry.array_key.rsplit('_', 1)[1]
    frame = array_rows(entry.source_path, entry.array_key)
    X = frame[value_columns(frame)].to_numpy(dtype=float)
    T = array_temperature(entry.source_path, entry.array_key)
    json_path = str(Path(entry.source_path).with_name('ising_chi_sweep.json'))
    match = sweeps[sweeps.source_path.eq(json_path)
                   & sweeps.method.eq(method) & sweeps.metric.eq('chi')
                   & np.isclose(sweeps['T'], T, rtol=0, atol=1e-10)]
    if len(match) != 1:
        raise ValueError(f'Expected one matching JSON measurement for {entry.source_path}, {method}, {T}')
    endpoint_records.append(dict(seed=match.iloc[0]['seed'], method=method, T=T,
        mean_abs_m=float(np.abs(X.mean(axis=1)).mean()), chi=float(match.iloc[0]['value'])))
endpoints = pd.DataFrame(endpoint_records)
print(endpoints)
fig, ax = plt.subplots(figsize=(6, 4))
for method, group in endpoints.groupby('method'):
    ax.scatter(group['mean_abs_m'], group['chi'], label=method.upper())
ax.set(xlabel='Mean absolute magnetization |m| from NPZ', ylabel='Saved susceptibility χ',
       title='Low-temperature observable comparison across seeds')
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / 'chi_vs_abs_m.pdf')
plt.show()
plt.close(fig)
```

With one effective disorder realization, these two observables describe the same saved endpoint sample set under the documented runner behavior. With multiple disorders, saved raw samples represent only realization zero while JSON metrics average disorders; do not treat them as the same aggregate.

## 12. Access diagnostic JSONs and summary TXT files

```python
print(pd.read_sql_query(
    "SELECT path FROM archived_files WHERE extension='.json' AND path LIKE '%diagnos%'", conn))
summary_paths = pd.read_sql_query(
    "SELECT path FROM archived_files WHERE extension='.txt' AND path LIKE '%summary%'", conn)
print(summary_paths.to_string(index=False))
if not summary_paths.empty:
    chosen = summary_paths.iloc[0]['path']
    raw = conn.execute('SELECT content FROM archived_files WHERE path=?', (chosen,)).fetchone()[0]
    print(raw.decode('utf-8', errors='replace'))
```

The same `json.loads(content)` approach works for diagnostic JSON, but its field names and nesting differ from sweep JSON. Inspect its keys before choosing variables. A raw-byte archive does not automatically turn a summary table into individual runs: you need a format-specific parser before plotting ladder/kitew/sdk/sdw text results. Preserve whether reported values are geometric means, arithmetic means, standard deviations, or SEMs, and whether learning rates were selected per cell/seed. Never invent seed-level observations from aggregate tables.

## 13. Optional: export a plotting DataFrame

```python
sweeps.to_csv(FIGURES / 'sweep_plot_data.csv', index=False)
endpoints.to_csv(FIGURES / 'endpoint_plot_data.csv', index=False)
conn.close()
```

These exports are convenience views. Keep the SQLite database as the common archive; retain source paths so figures remain traceable.

## Practical checks and limitations

- Start by inspecting `npz_catalog`. If an array's imported row count differs from its documented shape, rerun its import before plotting.
- Plot from one experiment with matched settings. The short importer did not recover missing hyperparameters from launch scripts, nor normalize all result formats. Missing LR, architecture, clipping, or schedule values cannot be inferred from today's defaults.
- The NPZ importer commits chunks separately. It can leave a partially imported array if interrupted; rerunning deletes and rebuilds the selected source/key rows. Avoid reading while that import is running.
- Reading one selected array at a time is appropriate for your documented 5×5 sample sizes. For a much larger array, use `pd.read_sql_query(..., chunksize=1000)` and accumulate statistics rather than concatenating all chunks.
- SQL value columns were created with REAL affinity. Floating-point spins and ordinary loss histories are suitable for plotting; original dtype information is in the catalog. Large integers may lose exact precision in these unpacked columns. The original NPZ bytes in `archived_files` remain the fidelity-preserving copy.
- Low/high-temperature samples do not recover intermediate-temperature configurations, and missing loss histories cannot be reconstructed from JSON metric curves.
- The examples were validated against synthetic fixtures matching the scripts and documented layouts. They have not been run against your Mac database; use the printed inventory to confirm your exact paths and array keys.


# **Code Guide**  

##
### Guide Description  
This guide is meant to be used to quickly ramp up the code usage, touching on basic problem formulation but assumes you know most of the background on overall project direction. For more description of the problem, I have another README file for Sergio's code that explains the project background.  

### Project Description


This project trains neural networks to sample spin configurations according to their Boltzmann probabilities. It compares two GFlowNet training methods—Trajectory Balance (TB) and the implementation’s Kolmogorov-Criterion method (KC)—with Markov chain Monte Carlo (MCMC).

The main program runs temperature sweeps, calculates physical observables, and saves results. A separate plotting program turns those results into figures.

## Project structure

The code has been split into modules without intentionally changing the algorithms.

| Module | Contents |
|---|---|
| `physics/env.py` | Physical systems, interaction pairs, couplings, and energy calculations |
| `physics/observables.py` | Magnetization, overlap moments, susceptibility, and Binder cumulants |
| `models/policy.py` | The neural network that chooses spin values |
| `models/gflownet.py` | Shared training/sampling code, TB and KC classes, and batch-size scheduling |
| `samplers/mcmc.py` | Metropolis and Wolff samplers, plus autocorrelation diagnostics |
| `spin_glass_lib.py` | Compatibility imports and constants retained for existing scripts |
| `run_tc_sweep.py` | Command-line interface, experiment coordination, checkpointing, and data saving |
| `make_plots.py` | Plotting commands that read saved results and produce PDFs |
| `compute_tc.py` | Optional script that summarizes saved temperature estimates across seeds |
| `sk_exact_enumeration_D10.py` | Optional exact-enumeration reference for small SK systems |
| `multi_N_crossing_bootstrap.py` | Optional comparison of Binder curves between different SK sizes |

Each of `physics`, `models`, and `samplers` contains an `__init__.py` file.

Run commands from the project root so Python can find these packages.

The compatibility module imports definitions from the new modules. Those modules should not import from `spin_glass_lib.py`, because doing so can create circular imports.

Some default hyperparameter constants are temporarily duplicated between the compatibility module and the GFlowNet module. This does not affect training becuase the default hyperparams are overwritten in command line calls, but consolidating them remains a cleanup task. 

## Physics models

### Spins and energy

A configuration contains `D` spins. Each spin is either `-1` or `+1`.

The energy is

$$
E(s)=-\sum_{(i,j)\in\mathcal E}J_{ij}s_i s_j.
$$

There is no external magnetic field.

Positive couplings favor aligned spins; negative couplings favor opposite spins. At positive temperature `T`, the target probability is

$$
p_T(s)=\frac{\exp[-E(s)/T]}{Z(T)},
\qquad
Z(T)=\sum_s\exp[-E(s)/T].
$$

The Boltzmann constant is set to one. Lower temperature puts more probability on low-energy configurations.

The neural sampler uses the unnormalized reward

$$
R(s)=\exp[-E(s)/T].
$$

Training calculates its logarithm rather than directly exponentiating large energy values.

### Supported models

| Model | Structure | Couplings |
|---|---|---|
| `ising` | An `N × N` square grid containing `D=N²` spins | Uniform `J=+1` |
| `ea` | An `N × N` square grid containing `D=N²` spins | Independent `J=-1` with probability `p_af`, otherwise `+1` |
| `sk` | A fully connected system containing `D` spins | Independent `J=-1/√D` with probability `p_af`, otherwise `+1/√D` |

Grid models use periodic boundaries: the grid wraps around both directions.

Sites are flattened in row-major order:

```python
i = row * N + column
```

For the grids used here, with `N ≥ 3`, the interaction builder stores the right and downward bond from every site. This gives `2N²` interactions, each included once in the energy sum.

The current construction needs separate consideration for `N=1` or `N=2`, where wrapping can produce self-interactions or duplicate undirected interactions.

SK stores every pair `i < j`, giving `D(D-1)/2` interactions. Its couplings are bimodal, not Gaussian.

**Size arguments have different meanings:**

- `--N 5` means a **5×5 grid with 25 spins** for Ising or EA.
- `--D 10` means **10 total spins** for SK.

### Disorder realizations and random seeds

A disorder realization is one fixed draw of EA or SK couplings.

The runner constructs its coupling seed as

```python
J_seed = seed * 1000 + d
```

where `d` ranges from zero to `n_disorder - 1`.

For each realization:

- The couplings remain fixed across temperatures.
- TB, KC, and MCMC use the same physical environment.
- Different outer seeds normally produce different disorder sets.

The runner also seeds NumPy and PyTorch with the outer `--seed`. Therefore, for EA/SK, changing the outer seed changes both disorder and training/sampling randomness.

Ising has no random couplings. Its effective disorder count is always one; additional outer seeds provide independent training and sampling runs.

If using more than 1,000 disorder realizations per outer seed, revise the coupling-seed convention to avoid overlap between adjacent outer seeds.

## GFlowNet background

A GFlowNet learns to construct objects through a sequence of decisions so that completed objects appear in proportion to their rewards.

Here, the completed object is a spin configuration.

The policy assigns spins in a fixed order:

1. Start with every site unassigned.
2. Assign site 0.
3. Assign site 1.
4. Continue until all `D` sites are assigned.

Action `0` assigns spin `-1`; action `1` assigns spin `+1`.

Every trajectory contains exactly `D` decisions. The model does not choose the next site or learn a stopping action.

Because the assignment order is fixed, each partial configuration has one parent. Each completed configuration has one construction path. Its backward probability is therefore one, giving

```python
LOG_PB_FIXED = 0.0
```

The physical reward is evaluated after the complete configuration has been constructed.

The goal is to learn a distribution, not just find one minimum-energy configuration.

Independent neural rollouts are independent draws from the learned policy, conditional on fixed network weights. They approximate the physical distribution only to the extent that training succeeds.

## Neural network inputs and outputs

`StepEmbedPolicy` is a multilayer perceptron shared across all assignment steps.

Let:

- `B` be the batch size.
- `D` be the number of spins.
- `t` be the next site to assign.

| Quantity | Shape | Description |
|---|---|---|
| Partial spins | `(B, D)` | Assigned spins are `-1/+1`; unassigned positions are zero |
| Assignment mask | `(B, D)` | One for assigned positions, zero otherwise |
| Step argument | Python integer | The same next-site index for the entire batch |
| Learned step embedding | `(B, 16)` | Embedding for the current assignment step |
| Grid coordinates | `(B, 2)` | Ising/EA only: normalized row and column of the next site |
| Concatenated features | `(B, 2D+18)` for grids; `(B, 2D+16)` for SK | Input to the first linear layer |
| Output logits | `(B, 2)` | Scores for assigning `-1` or `+1` |
| Action probabilities | `(B, 2)` | Softmax of the logits |
| Complete action sequences | `(B, D)` | Integer actions for all trajectories |
| Complete spin configurations | `(B, D)` | Terminal `-1/+1` spins |

For a 5×5 grid:

```text
D = 25
features = 25 spins + 25 mask entries + 16 embedding values + 2 coordinates
         = 68
```

With the default batch size, each policy call maps `(16, 68)` features to `(16, 2)` logits.

**The outputs are logits, not one-hot encodings.**

The network contains three hidden linear layers of width 128, each followed by GELU, and a final linear layer producing two logits.

The step-embedding table has shape `(D, 16)`. Grid coordinates are stored in a `(D, 2)` buffer using `(row/N, column/N)`.

The input does not explicitly include the temperature or coupling matrix. Separate models are trained for each temperature and disorder realization, with optional weight transfer between temperatures.

## Shared training and sampling

### `BatchedGFlowNetBase`

This class implements the machinery shared by TB and KC:

- Construct the policy.
- Load initial policy weights when requested.
- Define exploration and reward schedules.
- Sample complete trajectories.
- Replay recorded actions with gradients.
- Run optimizer iterations.
- Generate evaluation samples.

TB and KC provide their own loss functions and optimizers.

This wrapper is not a `torch.nn.Module`. Its `train()` method performs optimization; it does not merely switch a network into training mode.

### One training iteration

Each iteration:

1. Calculates the current schedule values.
2. Samples `B` complete trajectories without gradients.
3. Replays those trajectories with gradients enabled.
4. Calculates terminal energies and the loss.
5. Clears gradients and backpropagates.
6. Clips the policy gradient norm to 1.
7. Takes an Adam optimizer step.

Sampling advances all trajectories together, one spin position at a time. Each step makes one network call for the whole batch.

Replay returns:

| Output | Shape |
|---|---|
| Trajectory log probabilities | `(B,)` |
| Mean per-step entropies | `(B,)` |
| Terminal spins | `(B, D)` |

The environment maps terminal spins to `(B,)` energies. Each method produces one scalar loss.

### Exploration and schedules

Training samples actions from


$$q(a\mid s)=(1-\epsilon)\pi_\theta(a\mid s)+\epsilon/2.$$  


The loss evaluates the learned policy probability `πθ`, rather than the exploration mixture probability.

| Setting | Default behavior |
|---|---|
| Policy learning rate | `1e-3` |
| Training batch | 16 trajectories |
| Reward multiplier | Approximately 0.3 at the beginning, increasing to 1 |
| Exploration probability | Approximately 0.8 initially, decreasing to 0.05 |
| Entropy coefficient | Approximately 0.05 initially, decreasing to zero |
| Entropy annealing duration | First 80% of training |
| Evaluation exploration | Zero |

The implementation calls the reward multiplier `beta`, but physical inverse temperature is `1/T`.
  

Training uses

```python
logR = beta * (-energy / T)
```

so it gradually changes the training target toward the requested Boltzmann distribution. It does not multiply policy logits by `beta`.

Schedules use `iteration / n_iter`, starting at iteration one. Consequently, initial values are slightly beyond their nominal starting values.

Increasing `n_iter` changes the schedule as well as the update count. A 3,000-iteration run is not simply an identical 1,500-iteration run followed by extra updates.

### Trajectory Balance

`TBGFlowNet` learns the policy and a scalar `log_Z`.

Its residual is

$$
r=\log Z_\theta+\log P_F(\tau)-\log R(s),
$$

and its loss is

$$
L_{TB}=\operatorname{mean}(r^2)
-\lambda_H\operatorname{mean}(H).
$$

The backward-log-probability term is zero for this fixed-order construction.

`log_Z` starts at `D log(2)` unless overridden. It uses a separate Adam parameter group with learning rate

```python
LR_NN * lr_logz_mult
```

The default multiplier is 50, giving a normalizer learning rate of 0.05.

Gradient clipping applies to policy parameters; it does not include the separate `log_Z` parameter.

### KC

`KCGFlowNet` avoids fitting `log_Z`. It calculates

$$
\delta=\log P_F(\tau)-\log R(s)
$$

and minimizes

$$
L_{KC}=\operatorname{Var}(\delta)
-\lambda_H\operatorname{mean}(H).
$$

The implementation uses population variance:

```python
residual.var(unbiased=False)
```

At the target distribution, the residual is the same for every trajectory: `-log Z`. Some original comments incorrectly give that constant a positive sign.

The method computes variance directly; it does not allocate all trajectory pairs.

**Use a training batch larger than one.** With one trajectory, the variance is identically zero and supplies no balance-loss learning signal.

TB and KC loss magnitudes should not be compared directly as a measure of sampling quality.

## MCMC sampling

`run_mcmc` builds a fresh chain at each temperature.

With `--mcmc_method auto`:

- Ising uses Wolff cluster updates.
- EA and SK use single-spin Metropolis updates.

After warm-up, the chain performs `mcmc_thin` updates between retained configurations.

### Metropolis

Each update proposes flipping one randomly chosen spin.

It calculates the local energy change

$$
\Delta E=2s_i\sum_jJ_{ij}s_j.
$$

The flip is accepted automatically when `ΔE ≤ 0`, otherwise with probability `exp(-ΔE/T)`.

A Metropolis update is **one proposed spin flip**, not a complete lattice sweep.

### Wolff

Each update grows and flips a cluster of aligned spins.

The addition probability for uniform `J=1` is

$$
p_{\mathrm{add}}=1-\exp(-2/T).
$$

This implementation is valid only for the ferromagnetic Ising model. Forcing Wolff on EA or SK is rejected.

One Wolff update is a whole cluster flip. Its warm-up and thinning counts therefore represent different work from the same counts under Metropolis.

### Diagnostics

| Diagnostic | Meaning |
|---|---|
| `acceptance_rate` | Accepted fraction of Metropolis proposals, including warm-up counters |
| `mean_cluster_fraction` | Mean Wolff cluster size divided by `D`, including recorded warm-up clusters |
| `tau_int` | Estimated integrated autocorrelation time of the post-warm-up energy trace |

`NaN` acceptance for Wolff is expected. Cluster fraction is correspondingly `NaN` for Metropolis.

When diagnostics are enabled, energy is recorded after every raw post-warm-up update. The trace length is

```python
n_eval * mcmc_thin
```

`integrated_autocorr_time` uses an FFT and an automatic window. Its result measures energy correlation in raw update units. It does not prove equilibration or good mixing of every observable.

Thinning and random pairing do not repair a chain that has not equilibrated.

## Physical observables

### Ising susceptibility

For each configuration,

$$
m=\frac{1}{D}\sum_i s_i.
$$

The code calculates

$$
\chi=\frac{D}{T}
\left(\langle m^2\rangle-\langle|m|\rangle^2\right).
$$

This is an absolute-magnetization finite-size convention. It differs from a formula using `⟨m⟩²`.

Use the same definition when comparing with another implementation.

### Spin-glass overlap

Two configurations from the same disorder realization define

$$
q_{ab}=\frac{1}{D}\sum_i s_i^{(a)}s_i^{(b)}.
$$

The code randomly permutes configurations and forms disjoint pairs. With 800 configurations, it normally calculates 400 overlaps.

It then computes

$$
\chi_{SG}=D\langle q^2\rangle
$$

and

$$
U_4=1-\frac{\langle q^4\rangle}
{3\langle q^2\rangle^2+10^{-12}}.
$$

The implemented spin-glass susceptibility has no extra temperature factor.

Susceptibility and Binder functions generate their pairings separately, so their finite-sample moments need not come from identical pairs.

### Temperature estimates

`peak_estimate` finds the largest susceptibility and uses neighboring points for an interpolated peak position when possible.

`crossover_from_binder` finds the interval where one Binder curve has its largest absolute slope.

**The saved `Tcross_*_binder` value is a steepest-slope crossover, not an intersection of curves from two sizes.**

A finite 5×5 system has rounded behavior. These estimates are finite-system diagnostics, not automatically thermodynamic critical temperatures.

## Function reference

| Definition | What it does |
|---|---|
| `build_neighbour_list(N)` | Returns wrapped grid interaction pairs |
| `build_all_pairs(D)` | Returns all fully connected pairs with `i < j` |
| `SpinEnv` | Stores the system, couplings, neighbor lists, and cached interaction arrays |
| `energy_batch(X)` | Maps Torch configurations `(B,D)` to energies `(B,)` |
| `energy_fast_np(x)` | Maps one NumPy configuration `(D,)` to a scalar energy |
| `local_delta_energy(x,i)` | Calculates the energy change for flipping site `i` |
| `StepEmbedPolicy.forward` | Maps partial configurations, masks, and a step index to `(B,2)` logits |
| `_sample_actions_batch` | Samples integer action sequences `(B,D)` without gradients |
| `_replay_batch` | Recalculates log probabilities, entropy, and final spins for fixed actions |
| `train` | Runs optimizer iterations; returns whether training completed |
| `sample(n)` | Returns a list of `n` NumPy spin vectors, each `(D,)` |
| `state_dict` | Returns policy weights only; excludes optimizer and TB normalizer |
| `make_batch_size_schedule` | Returns the callable used to interpolate training batch sizes |
| `run_chain` | Runs warm-up and retained MCMC sampling; optionally returns an energy trace |
| `run_mcmc` | Chooses a sampler and optionally returns diagnostics |
| `integrated_autocorr_time` | Estimates autocorrelation time for a one-dimensional trace |
| `magnetization_susceptibility` | Returns Ising susceptibility and mean absolute magnetization |
| `spin_glass_overlap_moments` | Returns overlap samples and their second/fourth moments |
| `spin_glass_susceptibility` | Returns spin-glass susceptibility and mean absolute overlap |
| `binder_cumulant_sg` | Returns the overlap Binder cumulant |
| `susceptibility` | Chooses the appropriate susceptibility calculation |
| `order_parameter_samples_for_hist` | Returns magnetizations or overlaps for histogramming |
| `run_sweep_for_disorder` | Runs temperatures for one fixed coupling realization |
| `save_checkpoint` / `load_checkpoint` | Save and restore completed-temperature progress |
| `load_seed_jsons` | Loads matching result files from `seed_*` subdirectories |
| `stack_and_average` | Calculates pointwise means and SEM across outer seeds |

Legacy tuple-based environment methods remain available, including `root`, `step`, `encode`, `energy`, and `log_reward`. The current batched training path does not use them.

## Installation

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch numpy matplotlib
```

Check the installation:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('Apple GPU available:', torch.backends.mps.is_available())"
```

Record a working environment:

```bash
python -m pip freeze > requirements-lock.txt
```

Activate the environment again in each new terminal:

```bash
source .venv/bin/activate
```

## Command-line usage

The general command is:

```bash
python run_tc_sweep.py --model MODEL --T_min LOW --T_max HIGH [OPTIONS]
```

The required arguments are `--model`, `--T_min`, and `--T_max`.

### Model and temperature arguments

| Argument | Default | Meaning |
|---|---|---|
| `--model` | Required | `ising`, `ea`, or `sk` |
| `--N` | `5` | Grid side length for Ising/EA; total spins are `N²` |
| `--D` | `5` | Total spins for SK |
| `--p_af` | `0.5` | Probability of a negative EA/SK coupling |
| `--T_min` | Required | Lowest temperature; use a positive value |
| `--T_max` | Required | Highest temperature |
| `--n_T` | `14` | Number of evenly spaced temperature points |

`--p_af` does not change Ising couplings. Changing it away from 0.5 changes the mean SK coupling, so the physical interpretation also changes.

Use at least two temperatures for the slope calculation; a meaningful temperature study generally needs substantially more.

### Training arguments

| Argument | Default | Meaning |
|---|---|---|
| `--methods` | `TB,KC,MCMC` | Comma-separated methods to run |
| `--n_iter` | `1500` | Training updates per neural method, temperature, and disorder realization |
| `--batch_start` | `16` | Initial training batch-size schedule endpoint |
| `--batch_end` | `16` | Final training batch-size schedule endpoint |
| `--entropy_coef0` | `0.05` | Initial entropy regularization coefficient |
| `--entropy_anneal_frac` | `0.8` | Fraction of training over which entropy regularization decreases to zero |
| `--lr_logz_mult` | `50` | TB normalizer learning-rate multiplier |
| `--init_log_z_override` | None | Explicit initial TB log normalizer |
| `--warm_start` | Off | Transfer policy weights and TB normalizer between temperatures |
| `--device` | `cpu` | Device for neural computation |

The policy learning rate, hidden widths, embedding dimension, and exploration/reward schedule endpoints are currently module constants rather than CLI arguments.

With warm start disabled, temperatures run low to high with newly initialized models.

With warm start enabled, temperatures run high to low. Policy weights and TB `log_Z` transfer, but optimizer state does not. Training schedules restart at each temperature.

An explicit `--init_log_z_override` takes precedence over the transferred normalizer.

### Sampling and randomness arguments

| Argument | Default | Meaning |
|---|---|---|
| `--n_eval` | `800` | Retained configurations per method and temperature |
| `--n_disorder` | `1` | EA/SK coupling realizations per outer seed |
| `--seed` | `0` | Outer training/sampling seed and coupling-seed prefix |
| `--mcmc_warmup` | `50000` | Raw MCMC updates before retention |
| `--mcmc_thin` | `50` | Raw updates between retained configurations |
| `--mcmc_method` | `auto` | `auto`, `metropolis`, or `wolff` |
| `--mcmc_diagnostics` | On | Calculate energy-based MCMC diagnostics |
| `--no_mcmc_diagnostics` | Not selected | Disable those diagnostics |

`n_eval` is independent of the neural training batch size.

Increasing `n_eval` improves sampling precision but does not necessarily fix training bias or MCMC equilibration.

Ising always uses one effective disorder realization.

### Saving and logging arguments

| Argument | Default | Meaning |
|---|---|---|
| `--outdir` | `./outputs` | Directory for this run’s files |
| `--no_resume` | Off | Ignore an existing checkpoint and start fresh |
| `--log_loss` | Off | Record selected neural training-loss histories |
| `--log_loss_every` | `25` | Loss-recording interval |
| `--log_loss_T_indices` | `0,-1` | Temperature-grid indices to record |
| `--disagree_tol` | `0.35` | Relative susceptibility difference from MCMC to flag |

Negative temperature indices count from the end. Loss logging also records the first and final iteration.

The disagreement threshold is a warning criterion, not a statistical significance test.

### Quick smoke test

This checks imports, training, sampling, and writing outputs. Its results are not converged physics estimates.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 2 --n_iter 10 --n_eval 50 --n_disorder 1 --mcmc_warmup 100 --mcmc_thin 1 --device cpu --no_resume --outdir ./smoke_test_refactor
```

`--no_resume` ensures repeated tests execute again.

### Short timing test

This command took **6.735 seconds** on the measured M1 Pro setup:

```bash
time OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 2 --n_iter 100 --n_eval 200 --n_disorder 1 --mcmc_warmup 1000 --mcmc_thin 10 --device cpu --outdir ./timing_cpu_100
```

### Full 5×5 temperature sweep

This command took **8 minutes 55.62 seconds** after refactoring:

```bash
time OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 14 --n_iter 1500 --n_eval 800 --n_disorder 1 --seed 0 --mcmc_warmup 50000 --mcmc_thin 50 --device cpu --log_loss --outdir ./outputs/ising_5x5_refactored/seed_0
```

### Longer training comparison

This preserves the same physics and evaluation settings but doubles neural training iterations:

```bash
time OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_tc_sweep.py --model ising --N 5 --methods TB,KC,MCMC --T_min 2 --T_max 3 --n_T 14 --n_iter 3000 --n_eval 800 --n_disorder 1 --seed 0 --mcmc_warmup 50000 --mcmc_thin 50 --device cpu --log_loss --outdir ./outputs/ising_5x5_iter3000/seed_0
```

Expected time is approximately **15–20 minutes**, extrapolated rather than measured.

Use a separate base directory so the plotting program does not average different training settings together.

### Multiple independent runs

Run sequential seeds with distinct output directories:

```bash
for seed in {0..9}; do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u run_tc_sweep.py \
    --model ising --N 5 --methods TB,KC,MCMC \
    --T_min 2 --T_max 3 --n_T 14 \
    --n_iter 1500 --n_eval 800 --n_disorder 1 \
    --seed "$seed" --mcmc_warmup 50000 --mcmc_thin 50 \
    --device cpu --log_loss \
    --outdir "./outputs/ising_5x5_refactored/seed_${seed}" \
    > "ising_5x5_seed_${seed}.log" 2>&1 || break
done
```

Ten comparable seeds would take roughly **89 minutes sequentially** based on the measured full-run timing.

Select only unfinished seeds if some already completed. Completed runs delete their checkpoints; repeating their commands can retrain and overwrite results.

## Output data

### Where results are stored

Outputs go to the directory specified by `--outdir`, relative to the directory where the command is launched unless an absolute path is supplied.

Recommended layout:

```text
outputs/
  ising_5x5_refactored/
    seed_0/
    seed_1/
  ising_5x5_iter3000/
    seed_0/
```

The plotter expects per-seed folders named `seed_*`.

File tags are:

| Model | Tag |
|---|---|
| Ising | `ising` |
| EA | `ea` |
| SK with 10 spins | `sk_D10` |

Each run uses fixed filenames within its output directory. Never send concurrent jobs to the same directory.

### Saved files

| File | Contents |
|---|---|
| `{tag}_chi_sweep.json` | Aggregate curves, uncertainty arrays, temperature estimates, and diagnostic arrays |
| `{tag}_raw_data.npz` | Endpoint spin configurations, selected loss histories, and temperatures |
| `{tag}_tc_estimates.txt` | Human-readable settings and summaries |
| `{tag}_checkpoint.pt` | Intermediate progress; removed after successful completion |

No plots are generated by the sweep itself.

### JSON contents

The JSON contains:

- Model name, `N`, `D`, method list, temperature grid, and effective disorder count.
- `chi_tb_mean`, `chi_kc_mean`, and `chi_mcmc_mean`, as applicable.
- Corresponding `_sem` arrays.
- Susceptibility peak estimates such as `Tc_tb_chi_peak`.
- EA/SK Binder means, SEMs, and steepest-slope estimates.
- MCMC diagnostic means and SEMs.

Curve arrays have length `n_T`.

Within one run, means and SEMs are calculated across disorder realizations. With one realization, those SEM arrays are zero; this does not mean the observables have no sampling uncertainty.

The JSON is not a complete record of every setting. Preserve the command, logs, environment versions, and code revision.

**Grid metadata caveat:** the saved JSON `D` field comes from the runner’s `--D` argument even for grid models. Use `N²` for the actual Ising/EA spin count.

Diagnostic `NaN` values may appear in the JSON. Strict JSON readers may need to handle those explicitly.

### Raw NPZ contents

Example keys include:

```text
T_grid
samples_lowT_tb
samples_lowT_tb_T
samples_highT_tb
samples_highT_tb_T
samples_lowT_kc
samples_highT_kc
samples_lowT_mcmc
samples_highT_mcmc
loss_tb_Tidx0
loss_tb_Tidx0_T
```

Sample arrays have:

```text
shape: (n_eval, D)
dtype: float32
values: -1 or +1
```

Loss arrays have:

```text
shape: (n_records, 2)
dtype: float64
columns: iteration, loss
```

**Retention limits:**

- Final raw samples are saved only for the lowest and highest temperatures.
- They come only from disorder realization zero.
- Loss histories also come only from realization zero, at requested temperature indices.
- Intermediate-temperature samples are not retained in the final NPZ.
- The completed run does not export every trained model for later reuse.

The runner temporarily retains more first-disorder samples during execution and checkpointing than it ultimately writes to the final NPZ.

### Approximate file sizes

For the measured configuration:

```text
25 spins
800 samples
3 methods
2 endpoint temperatures
float32 values
```

The raw spin arrays alone require

```text
25 × 800 × 3 × 2 × 4 bytes = 480,000 bytes
```

That is approximately **0.48 MB**, before archive headers, temperatures, and loss records.

With 1,500 iterations, loss logging every 25 iterations, two logged temperatures, and two neural methods, the loss arrays add approximately **3.9 KB**.

| Artifact | Approximate scale for this 5×5 setup |
|---|---|
| Raw NPZ | Around half a megabyte |
| JSON | Usually kilobytes to tens of kilobytes |
| Text summary | Usually a few kilobytes |
| Checkpoint | Variable; can exceed final data because it retains intermediate samples and state |
| Plot PDFs | Variable with figure content |

These are estimates except for the calculated array payloads. Inspect actual sizes with:

```bash
du -h ./outputs/ising_5x5_refactored/seed_0/*
```

Final raw-sample storage grows approximately with `D × n_eval × number_of_methods`. Increasing `n_T` grows aggregate curves and may grow loss logs, but final raw configurations still contain only two temperatures.

### What makes two runs different?

| Changed setting | What changes |
|---|---|
| `seed` | Training/sampling randomness; also EA/SK couplings |
| `N` or `D` | Physical system size and network input dimensions |
| `p_af` | EA/SK coupling distribution |
| Temperature range/grid | Which physical distributions are sampled |
| `n_iter` | Training budget and schedule progression |
| `batch_start/end` | Number of trajectories per update and stochastic loss behavior |
| `n_eval` | Number of evaluation configurations and estimator precision |
| MCMC warm-up/thinning | Chain work and retained-sample correlation |
| `n_disorder` | Number of coupling realizations averaged within a run |
| `warm_start` | Temperature order and transfer of learned weights |
| `methods` | Which algorithms execute; also changes subsequent random-number consumption |
| Device/backend | Performance and potentially numerical/reproducibility behavior |

Keep different experimental settings in different base directories. Seeds intended for averaging should otherwise use matching settings.

## Plotting

Create aggregate curves:

```bash
python make_plots.py chi --base_dir ./outputs/ising_5x5_refactored --tag ising --model_kind ising --outdir ./figures/ising_5x5_refactored
```

Create endpoint histograms:

```bash
python make_plots.py distributions --seed_dir ./outputs/ising_5x5_refactored/seed_0 --tag ising --model_kind ising --outdir ./figures/ising_5x5_refactored
```

Create training-loss plots:

```bash
python make_plots.py loss --seed_dir ./outputs/ising_5x5_refactored/seed_0 --tag ising --outdir ./figures/ising_5x5_refactored
```

Create MCMC diagnostics:

```bash
python make_plots.py mcmc_diag --base_dir ./outputs/ising_5x5_refactored --tag ising --outdir ./figures/ising_5x5_refactored
```

On macOS:

```bash
open ./figures/ising_5x5_refactored
```

The plotter writes PDFs using a noninteractive backend.

`chi` averages saved mean curves across outer seeds and calculates SEM across those seeds. It does not propagate each run’s internal disorder SEM. With one outer seed, its SEM is zero.

The loader searches:

```text
base_dir/seed_*/{tag}_chi_sweep.json
```

It does not read a JSON directly inside `base_dir`. The distribution command can read a run directory directly through `--seed_dir`.

## Checkpointing and restarting

Progress is saved after a whole temperature point completes all requested methods.

Restart using the same command and output directory without `--no_resume`.

Important limitations:

- Interrupted neural training at the current temperature is discarded.
- Completed temperatures can be reused.
- Optimizer state is not saved.
- Random-number-generator states are not saved.
- Resumed runs are not guaranteed to match uninterrupted runs bit for bit.
- Changed argument signatures cause a warning and fresh execution.
- Source-code changes are not comprehensively detected.
- Final successful execution removes the checkpoint.

The runner handles `SIGTERM` and `SIGUSR1`. During neural training, it checks for interruption between updates. During MCMC, it may need to finish the current chain before responding.

A gracefully interrupted run exits with status zero, so exit status alone does not prove completion.

Only load trusted checkpoints: the current loader allows Python object deserialization.

## Parallelization and optimization

### Independent experiments

The current runner executes sequentially.

The simplest parallel work units are independent outer seeds. Give each worker:

- Its own seed.
- Its own output directory.
- A controlled CPU thread count.

Disorder realizations currently loop inside the runner. Parallelizing them requires additional coordination and aggregation.

Warm-started temperatures depend on previous weights and must remain ordered. Splitting non-warm-start temperatures is possible in principle but requires output merging and changes random-number consumption.

### CPU threading

The measured baseline uses:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Compiled library kernels may use threads, but these settings do not parallelize Python loops.

Small network operations can become slower with excessive threading. Benchmark one, two, and four threads rather than assuming more is better.

With multiple processes, keep total thread usage within available or allocated CPU resources.

### GPU execution

Test `--device mps` on Apple hardware or `--device cuda` on compatible NVIDIA hardware.

This moves neural computations and Torch energy calculations. MCMC remains CPU code.

The small network, batch size 16, and sequential assignment decisions may make GPU overhead dominate. Benchmark before paying for GPU access.

### Candidate improvements

These were suggested but are not part of the structural refactor:

1. Convert sampled actions directly to terminal spins instead of replaying the network during evaluation.
2. Batch fixed-action training replay across timesteps.
3. Compile array-based MCMC update loops.
4. Benchmark thread counts and larger batches.
5. Reduce diagnostic overhead only after establishing trustworthy sampling settings.

Replay batching requires an interface accepting a different step index per row. Sampling still depends on previous decisions.

## Running on a cluster

A cluster mainly helps run many seeds simultaneously. A single cluster CPU may not outperform the Mac.

Create a fresh Linux environment following the institution’s instructions. Do not copy the Mac virtual environment onto the cluster.

The following Slurm example runs ten seeds with at most four concurrent tasks. Adjust account, partition, Python setup, memory, and wall time for the institution.

Save as `run_ising_array.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=ising5
#SBATCH --array=0-9%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH --signal=USR1@300
#SBATCH --output=logs/ising_%A_%a.out
#SBATCH --error=logs/ising_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Load the institution's Python module here if required.
source .venv/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

seed="$SLURM_ARRAY_TASK_ID"

srun python -u run_tc_sweep.py \
  --model ising --N 5 --methods TB,KC,MCMC \
  --T_min 2 --T_max 3 --n_T 14 \
  --n_iter 1500 --n_eval 800 --n_disorder 1 \
  --seed "$seed" \
  --mcmc_warmup 50000 --mcmc_thin 50 \
  --device cpu --log_loss \
  --outdir "./outputs/ising_5x5_cluster/seed_${seed}"
```

Submit from the project root:

```bash
mkdir -p logs
sbatch run_ising_array.sbatch
squeue -u "$USER"
```

Create `logs` before submission because Slurm opens log files before executing the script.

The early signal gives the program an opportunity to checkpoint. Confirm signal delivery and restart behavior on the actual cluster. Checkpointing does not automatically resubmit the job.

Memory and time requests are starting allocations, not measured requirements.

If cluster jobs match the Mac’s nine-minute timing, ten seeds on four simultaneous workers would need roughly three waves, or 27 minutes of running time, plus queue delay and overhead.

Do not run production sweeps on login nodes. Install dependencies once using the institution’s approved workflow.

## Exact enumeration

The supplied exact script enumerates all `2ᴰ` configurations, then explicitly constructs every state-pair overlap.

At `D=10`, this is practical:

```bash
python sk_exact_enumeration_D10.py --D 10 --p_af 0.5 --T_min 0.2 --T_max 0.4 --n_T 2 --seed 0 --n_disorder 3 --outdir ./exact_D10
```

Match temperatures, coupling probability, and seeds when comparing with sampled SK results.

The pairwise implementation scales as `O(4ᴰ)`. One float64 overlap matrix requires:

| Spin count | Matrix storage |
|---|---|
| 10 | 8 MiB |
| 14 | 2 GiB |
| 25 | 8 PiB |

The code allocates several such matrices and temporaries. Blocking reduces peak memory but does not remove the total pairwise work.

A proposed Walsh–Hadamard/XOR-convolution replacement could reduce the overlap calculation to `O(D2ᴰ)` work and `O(2ᴰ)` memory. It has not been implemented or timed.

The exact helper’s returned `Z` is also shifted by its numerical-stability transformation. For the physical normalizer, preserve the original maximum log weight and calculate:

```python
logZ = original_logw_max + np.log(w.sum())
```

Normalized probabilities and overlap moments are unaffected by that shift.

## Validation and interpretation

A successful run proves that the code executes and saves results. It does not prove that either neural sampler or MCMC has converged.

Before drawing conclusions:

- Compare multiple outer seeds.
- Check sensitivity to training length and evaluation count.
- Inspect endpoint distributions and training histories.
- Check MCMC equilibration and observable-specific mixing.
- Validate small SK systems against exact enumeration.
- Keep observable definitions consistent.
- Separate finite-size crossovers from thermodynamic transition claims.

The optional `compute_tc.py` averages saved per-seed temperature estimates. That differs from locating a peak on a seed-averaged curve.

The optional bootstrap script studies intersections between different SK sizes. Its historical `--N_values` refers to SK spin counts, with directories such as `D10` and `D25`, rather than grid side lengths.



## References

- [Trajectory balance: Improved credit assignment in GFlowNets](https://arxiv.org/abs/2201.13259)
- [PyTorch CPU thread control](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html)
- [PyTorch multiprocessing best practices](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)
- [Slurm job arrays](https://slurm.schedmd.com/job_array.html)
- [Slurm batch options and signals](https://slurm.schedmd.com/sbatch.html)