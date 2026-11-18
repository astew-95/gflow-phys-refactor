# main.py — run from the project root (gflow-phys-refactor)
import argparse
import random
from pathlib import Path
import numpy as np
import torch
from models.gflownet import TBGFlowNet, KCGFlowNet
from physics.spin_system import SpinEnv
from utils.save_run_history import atomic_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ising", "ea", "sk"], default="ising")
    parser.add_argument("--N", type=int, default=3)
    parser.add_argument("--D", type=int, default=9)
    parser.add_argument("--T", type=float, default=2.27)
    parser.add_argument("--methods", nargs="+", choices=["TB", "KC"], default=["TB", "KC"])
    parser.add_argument("--n_iter", type=int, default=800)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--tilt_strength", type=float, default=1.0)
    parser.add_argument("--schedule", choices=["original", "fixed"], default="original")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.T <= 0 or args.n_iter < 1 or args.batch < 2:
        parser.error("Require T > 0, n_iter >= 1, and batch >= 2")

    args.outdir.mkdir(parents=True, exist_ok=True)
    classes = {"TB": TBGFlowNet, "KC": KCGFlowNet}

    for method in args.methods:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        size = {"D": args.D} if args.model == "sk" else {"N": args.N}
        env = SpinEnv(args.model, J_seed=args.seed, **size)
        model = classes[method]( env, args.T, tilt_strength=args.tilt_strength, schedule=args.schedule,)
        name = ( f"{args.model}_D{env.D}_{method}_T{args.T}" f"_tilt{args.tilt_strength}_{args.schedule}_seed{args.seed}" )
        path = args.outdir / f"{name}.json"
        status = "error"

        try:
            completed = model.train( args.n_iter, batch_size_fn=lambda it, total: args.batch, )
            status = "completed" if completed else "interrupted"
        
        except KeyboardInterrupt:
            status = "interrupted"
            raise

        finally:
            config = {**vars(args), "outdir": str(args.outdir)}
            atomic_json(path, { "config": config,"method": method, "D": env.D, "status": status, "curve": getattr(model, "training_history", []),})

        print(f"{method}: {status}; saved {path}")


if __name__ == "__main__":
    main()