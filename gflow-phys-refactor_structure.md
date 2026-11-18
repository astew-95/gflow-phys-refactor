# gflow-phys-refactor structure

```
gflow-phys-refactor/
├── about
│   ├── algorithm.pdf
│   ├── algorithm.tex
├── data
│   └── sergio_results.sqlite
├── experiments
│   ├── demos
│   │   └── demo_small_diamonds.py  # sergio's small diamonds demo
│   ├── notebooks                   # interactive notebooks exploring results
│   │   ├── gflow_ising_viz.ipynb
│   │   └── tilted_ising_viz.ipynb
|   └── scripts                     # these are some old experiments. 
│       ├── diamond_clip_sweep.py
│       ├── loop_big.py
│       ├── loop_gfn.py
│       ├── loop_grid.py
│       ├── main_diamond_clip.py
│       ├── main_diamond_clip_sweep.py
│       ├── tc_sweep.py
│       ├── test_loop_gfn.py
│       ├── tilt_5x5_probe.py
│       ├── tilt_early_probe.py
│       └── tilt_early_probe_save.py
├── models
│   ├── ffn.py
│   ├── gflownet.py
│   ├── mcmc.py
│   └── policy.py
├── physics
│   ├── observables.py
│   ├── save_spin_env.py
│   ├── sk_exact_enumeration_D10.py
│   ├── spin_glass_lib.py
│   └── spin_system.py
├── postprocessing
│   ├── compute_tc.py
│   ├── data_aggregate_to_sql.py
│   ├── diagnose_loops.py
│   ├── make_plots.py
│   ├── multi_N_crossing_bootstrap.py
│   ├── plot_loop_sdk.py
│   ├── summarize_kc_ising3.py
│   └── summarize_kc_ladder.py
├── results
│   ├── images
│   │   ├── small_diamond_figures
│   │   ├── tilted_small_figures
│   └── runs_output
...
├── scripts
│   └── run_local.sh
├── slurm
│   └── small_diamonds_sergio
│       ├── submit_diag_sdk.slurm
│       ├── submit_kc_kitew.slurm
│       ├── submit_loop_sdk.slurm
│       └── submit_loop_sdw.slurm
├── theory
│   ├── ising_exact.py
│   ├── kc_exact.py
│   └── loop_d8_exact_check.py
├── utils
│   ├── mcmc_utils.py
│   ├── misc.py
│   └── model_utils.py
└── .gitignore
```
