"""
spin_glass_lib.py

==============
Alexa's Comments: 

Computes the training reward from the energy of the completed configuration, after all spins are assigned.

Parents in the GFlowNet: 
        - GFlowNet always assigns spins in a fixed order: position 0, then 1, then 2, etc.
For example, [-1, +1, 0] can only come from [-1, 0, 0]

Inputs/ Outputs:
    For default Ising (N=5, D=25), each network call takes 
        
        (batch_size,68) features
            - 25 spin values: -1/+1 if assigned, 0 if unassigned.
            - mask (value 1 for spins that have been assigned in the trajectory. The trajectory is length 25 (5x5 grid). A zero in the mask means this grid position has not been assigned. 
            - size 16 embedding
            - 2 grid coordinates corresponding to grid location. 
        
        returns (batch_size,2) scores
            - The output is one-hot encoded for spins +1 or -1 for batch size of 16. 
            - training defaults to batch_size=16


Note: after refactor, spinclasslib is just a compatibility layer for legacy implementations. It's contents have been moved to the appropriate libraries.

==================
Shared physics + sampling code for the critical-temperature sweeps.

# ------------------------------------------------------------------
# DOCSTRING CORRECTED -- the previous version of this header was
# stale and described an earlier iteration of this file, not the one
# below it:
#   - It said "drops KC" -- KCGFlowNet is fully implemented below.
#   - It described Ising/EA as a fixed 5x5 grid and SK as D=5 --
#     the actual sweeps this code has been run with go up to N=10
#     (Ising) and D=100 (SK). D/N are CLI-configurable (see
#     run_tc_sweep.py), not fixed at 5.
# Left uncorrected for a while, which is exactly the kind of drift
# worth watching for elsewhere in this file too. If you find another
# copy of this module with a docstring matching the OLD text below,
# that copy is more likely to be the genuinely older version:
#   "Trimmed / adapted from the notebook prototype: drops KC, XY, and
#    exact enumeration... Ising: 5x5 periodic grid... SK: D=5
#    fully-connected..."
# ------------------------------------------------------------------

Provides:
  - SpinEnv: unified +-1 spin environment for Ising (grid), EA (grid,
    disordered), and SK (fully-connected, disordered) models.
  - TBGFlowNet / KCGFlowNet: batched, weight-tied-across-timesteps
    GFlowNet training, differing only in their loss (TB fits an
    explicit log_Z; KC eliminates it via a telescoping residual
    variance).
  - MetropolisHastings / WolffCluster + run_mcmc(): MCMC baselines
    (Wolff for uniform-J Ising only; Metropolis for EA/SK).
  - susceptibility estimators (ferromagnetic chi for Ising, spin-glass
    chi_SG for EA/SK) computed from *independent equilibrium samples*
    (the standard "two-replica overlap" trick -- no ground truth
    needed), plus binder_cumulant_sg().
  - Optional warm-start: initialize a policy at temperature T from the
    already-trained policy at the neighboring T in the sweep.

# ------------------------------------------------------------------
# LEGACY API NOTE: SpinEnv carries a second, older interface alongside
# the batched/tensor one actually used for training --
# root()/step_index()/step()/encode()/state_to_x()/energy()/
# log_reward(), a tuple-based single-trajectory representation. Cross-
# checked against every file currently known to import this module
# (run_tc_sweep.py, make_plots.py, sk_exact_enumeration_D10.py): NONE
# of them call this API. run_tc_sweep.py's actual training path goes
# through BatchedGFlowNetBase._sample_actions_batch/_replay_batch,
# which build spins/mask tensors directly and never touch
# root()/step()/encode(); reward uses energy_batch() (the vectorized
# version), never energy()/log_reward()/state_to_x().
# sk_exact_enumeration_D10.py reimplements SK coupling construction
# independently rather than calling SpinEnv at all.
# Each flagged method below is marked individually with why it looks
# unused. Marked here rather than removed, since "not called by any
# file currently known to import this module" isn't the same
# guarantee as "not called anywhere" -- an unseen notebook or script
# could still depend on it. Safe to delete once that's ruled out.
# ------------------------------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.policy import StepEmbedPolicy
from physics.spin_system import SpinEnv, build_neighbour_list, build_all_pairs
from physics.observables import ( magnetization_susceptibility, spin_glass_overlap_moments, spin_glass_susceptibility, binder_cumulant_sg,
    susceptibility, order_parameter_samples_for_hist,
)
from models.mcmc import (integrated_autocorr_time, MetropolisHastings, WolffCluster, run_mcmc,)
from models.gflownet import BatchedGFlowNetBase, TBGFlowNet, KCGFlowNet
from utils.model_utils import make_batch_size_schedule

# ---------------------------------------------------------------
# Hyperparameters (safe defaults; override via CLI in run_tc_sweep.py)
# ---------------------------------------------------------------
BATCH        = 16         # Default number of trajectories per training batch
LR_NN        = 1e-3       # Default learning rate
LR_LOGZ_MULT = 50.0       # multiplty the learning rate for Trajectory Balance by LR_LOGZ_MULT (lr TB is LR_NN*LR_LOGZ_MULT)
EPS_START    = 0.8        # Initial exploration mixture weight
EPS_END      = 0.05       # Final exploration mixture weight
BETA_START   = 0.3        # Initial reward multiplier in logR = beta * (-E/T)
BETA_END     = 1.0        # Final reward multiplier: full thermal target
ENTROPY_COEF = 0.05       # Initial entropy-bonus weight, gradually reduced to zero during training
MCMC_WARMUP  = 50_000     # Run MCMC_WARMUP number of update steps before sampling with MCMC, let system settle
MCMC_THIN    = 50         # save MCMC config every MCMC_THIN updates 
LOG_PB_FIXED = 0.0        # For backward probabilities, this graph has unique backward parents. So backward probabilities are 1. 



