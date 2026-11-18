"""

Summary table for the kc_exact.py 3x3 Ising grids (outputs_kc_ldsd3/): 

final exact KL(q||p) and TV, best lr per cell, geometric mean (KL) / mean (TV) over init seeds.

Usage: python summarize_kc_ising3.py [ldsd3|ising3]

"""
import glob, json, sys
from collections import defaultdict
import numpy as np

rows = [json.load(open(f)) for f in glob.glob(f"outputs_kc_{sys.argv[1] if len(sys.argv) > 1 else 'ldsd3'}/*.json")]
cell = defaultdict(list)

for r in rows:
    cell[(r["policy"], r["grad"], r["dag"], r["method"], r["weights"], r["T"], r["J"])].append(r)

keys = sorted({k[:5] for k in cell}, key=lambda k: (k[1] != "exact", k[0] != "table", k[2], k[3], k[4]))
nseed = len({r["J"] for r in rows})
Ts = sorted({r["T"] for r in rows}, reverse=True)
print(f"{len(rows)} runs. 3x3 periodic Ising, final exact KL(q||p) [TV], best lr per (cell, seed)\n")
print(f"{'condition':15s}{'dag':6s}{'objective':13s}{'weights':13s}" + "".join(f"{'T=' + str(T):>20s}" for T in Ts))
for k in keys:
    line = f"{k[0] + '/' + k[1]:15s}{k[2]:6s}{k[3]:13s}{k[4]:13s}"
    for T in Ts:
        kl, tv = [], []
        for J in (0, 1, 2):
            rs = cell.get(k + (T, J))
            if rs:
                b = min(rs, key=lambda r: r["final"]["kl_qp"])
                kl.append(max(b["final"]["kl_qp"], 1e-12)); tv.append(b["final"]["tv"])
        line += f"{np.exp(np.mean(np.log(kl))):10.2e} [{np.mean(tv):.3f}]" + (f"{len(kl)}" if len(kl) < nseed else " ") if kl else f"{'-':>20s}"
    print(line)
    
print("\ntrailing digit = number of seeds finished, if incomplete")
