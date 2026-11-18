#!/usr/bin/env bash
set -euo pipefail
start_time=$(date +"%I:%M:%S %p on %Y-%m-%d ")
echo "Local run started at $start_time"

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" # root is one above this scripts directory, gflow-phys-refactor.
cd "$root"

# ------------------------------- local script, current working directory is gflow-phys-refactor --------------------------------------

seed="0"
out="$root/results/smoke_test"

python -u experiments/scripts/run_tc_sweep.py --model ising --N 3 --methods TB --T_min 1 --T_max 1 --n_T 14 --n_iter 300 --n_eval 80 --n_disorder 1 --seed "$seed" --device cpu --log_loss --outdir "$out/seed_$seed" 


# ---------------------------------------------------------------------
end_time=$(date +"%I:%M:%S %p on %Y-%m-%d ")
echo "Local run completed at $end_time" 