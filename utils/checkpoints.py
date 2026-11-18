import torch
import signal
# =====================================================
#  Signal handling for graceful preemption
# =====================================================
_STOP_REQUESTED = [False]


def _signal_handler(signum, frame):
    if not _STOP_REQUESTED[0]:
        print(f"\n[checkpoint] Caught signal {signum} -- will checkpoint and "
              f"exit as soon as it's safe to do so.", flush=True)
    _STOP_REQUESTED[0] = True


def _install_signal_handlers():
    signal.signal(signal.SIGTERM, _signal_handler)
    try:
        signal.signal(signal.SIGUSR1, _signal_handler)
    except (AttributeError, ValueError):
        pass


def save_checkpoint(path, payload):
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path):
    if not os.path.exists(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"WARNING: failed to load checkpoint at {path} ({e}); "
              f"starting fresh instead.", flush=True)
        return None