"""

Summary table for the kc_exact.py ladder (outputs_kc_ladder/): 

final exact KL(q||p), best learning rate per cell, geometric mean over J seeds. 

Usage: python summarize_kc_ladder.py

"""

import glob, json, os
from collections import defaultdict
import numpy as np

rows = [json.load(open(f)) for f in glob.glob("outputs_kc_ladder/*.json")]
cell = defaultdict(list)
for r in rows:
    tag = r["method"] + ("_unif" if r["weights"] == "uniform" else "")
    cell[(r["policy"], r["grad"], tag, r["D"], r["T"], r["J"])].append(r)

conds = [("table", "exact"), ("table", "sampled"), ("net", "exact"), ("net", "sampled")]
tags = ["ld", "basis", "basis_unif"]
Ds = sorted({r["D"] for r in rows}); Ts = sorted({r["T"] for r in rows}, reverse=True)
print(f"{len(rows)} runs. Final exact KL(q||p), best lr per (cell, J), geo-mean over J\n")
for T in Ts:
    print(f"T = {T}")
    print(f"{'condition':16s}{'objective':12s}" + "".join(f"{'D=' + str(D):>12s}" for D in Ds))
    for pol, grad in conds:
        for tag in tags:
            line, any_ = f"{pol + '/' + grad:16s}{tag:12s}", False
            for D in Ds:
                v = []
                for J in (0, 1):
                    rs = cell.get((pol, grad, tag, D, T, J))
                    if rs:
                        v.append(min(max(r["final"]["kl_qp"], 1e-12) for r in rs))
                line += f"{np.exp(np.mean(np.log(v))):12.2e}" if len(v) == 2 else f"{'-':>12s}"
                any_ |= bool(v)
            if any_:
                print(line)
    print()
