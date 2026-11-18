# utils/save_run_history.py
from contextlib import contextmanager
import json
import random
import numpy as np
import torch

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

# @contextmanager
# def preserve_rng():
#     py=random.getstate(); npstate=np.random.get_state(); th=torch.get_rng_state()
#     try: yield
#     finally:
#         random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(th)

@contextmanager
def preserve_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)