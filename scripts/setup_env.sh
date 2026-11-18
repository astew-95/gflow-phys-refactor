#!/usr/bin/env bash
# Save in the project root, then run: bash setup_env.sh
# Optional interpreter override: PYTHON=python3.12 bash setup_env.sh
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
trap 'echo "Setup failed. Fix the error above, then rerun this script." >&2' ERR

if [[ "$(uname -s)" != Darwin ]]; then
  echo "This script targets Apple Silicon macOS." >&2
  exit 1
fi

validate_python() {
  "$1" - <<'PY'
import platform
import sys
if platform.machine() != "arm64":
    sys.exit("Use native ARM64 Python, not Intel Python under Rosetta.")
if sys.version_info < (3, 11):
    sys.exit("Use Python 3.11 or newer; Python 3.12 is a reasonable choice.")
print(f"Python {platform.python_version()} ({platform.machine()})")
PY
} 

if [[ -e .venv || -L .venv ]]; then
  if [[ ! -f .venv/pyvenv.cfg || ! -x .venv/bin/python ]]; then
    echo "Existing .venv is invalid; move it aside before rerunning." >&2
    exit 1
  fi
else
  python_cmd="${PYTHON:-python3}"
  if ! command -v "$python_cmd" >/dev/null 2>&1; then
    echo "Install ARM64 Python first (for example: brew install python@3.12)." >&2
    echo "Then run: PYTHON=python3.12 bash setup_env.sh" >&2
    exit 1
  fi
  validate_python "$python_cmd"
  "$python_cmd" -m venv .venv
fi

validate_python .venv/bin/python
.venv/bin/python -m pip install --upgrade pip

# First run resolves compatible stable wheels; later runs reuse the snapshot.
# Delete this generated snapshot explicitly when you want to resolve new versions.
lock_file="requirements-macos-arm64.lock"
if [[ -f "$lock_file" ]]; then
  .venv/bin/python -m pip install --only-binary=:all: -r "$lock_file"
fi
.venv/bin/python -m pip install --only-binary=:all: torch numpy matplotlib pandas
.venv/bin/python -m pip check

.venv/bin/python - <<'PY'
import platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import pandas as pd

assert np.isclose(np.arange(4).sum(), 6)
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
fig.canvas.draw()
plt.close(fig)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
x = torch.ones((2, 2), device=device)
assert (x @ x).cpu().tolist() == [[2.0, 2.0], [2.0, 2.0]]
print(f"torch={torch.__version__}, numpy={np.__version__}, matplotlib={matplotlib.__version__}")
print(f"Verified device: {device}; macOS {platform.mac_ver()[0]}")
if device.type != "mps":
    print("MPS unavailable: CPU works. Check macOS/PyTorch compatibility for GPU use.")
PY

if [[ ! -f "$lock_file" ]]; then
  .venv/bin/python -m pip freeze > "$lock_file"
fi

if ! [[ -f .gitignore ]] || ! grep -qxF '.venv/' .gitignore; then
  printf '\n.venv/\n' >> .gitignore
fi

echo "Ready. Activate from the project root: source .venv/bin/activate"
echo "All subfolders share this environment. Select .venv/bin/python in your editor."
echo "Commit $lock_file for reuse with the same Python minor version and macOS architecture."
echo 'For GPU use on Apple M1 Pro chip: device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")'
