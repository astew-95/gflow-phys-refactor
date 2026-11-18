#!/usr/bin/env python3
"""
data_aggregate_to_sql.py.py
This script is for importing / aggregating all of sergio's data from previous runs. 

Located in gflow-phys/aggregate_plotting

All results files / folders I am accessing right now: 

gflow-phys/small_diamonds/results 
Contains: 
    - summary.txt files for exact_ladder, kitew, main 10x10, sdk, sdw
    - diagose_main.json and diagnose_sdk.json

    
gflow-phys/GFN_Thesis/outputs
Contains: 
- ising 5x5 refactored  (10 seed folders (eg seed_0), each containing ising_chi_sweep.json, ising_raw_data.npz, ising_tc_estimates.txt)
- smoke_test_refactor
- smoke_test_refactor_ea
- smoke_test_refactor_sk


gflow-phys/GFN_Thesis/results/runs
Contains: 
- smoke_test, smoke_test _refactor, timing_cpu_100, timing_cpu_1500 (each containing ising_chi_sweep.json, ising_raw_data.npz, ising_tc_estimates.txt)

Also these files: 
gflow-phys/GFN_Thesis/plot_inputs/timing_cpu_1500/seed_0/ contains ising_chi_sweep.json ising_raw_data.npz

"""
## Aggregate sergio's data into sql database for cross-plotting 

import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import io, hashlib

root = Path(__file__).resolve().parents[2]  # postprocessing/data_aggregate_to_sql.py

## Collects .json, .npz, and .txt data from subfolders in this folders list. Assumes they are relative to root.
COLLECT_FROM_FOLDERS = ["small_diamonds/results", "GFN_Thesis/outputs", "GFN_Thesis/results/runs", "GFN_Thesis/plot_inputs/timing_cpu_1500"
                        
                        # small_diamonds/code
                        "small_diamonds/code/outputs_kc_kitew", "small_diamonds/code/outputs_clip_lr_20261108_153232",
                        "small_diamonds/code/outputs_kc_ladder", "small_diamonds/code/outputs_loop_big",
                        "small_diamonds/code/outputs_loop_sdk","small_diamonds/code/outputs_loop_sdw",
                        
                        # small diamonds/outputs
                        "gflow-phys/small_diamonds/outputs/local_diamond_clip_20261111_143827"
                        ]
results_filename="sergio_results"

db = sqlite3.connect(root / f"aggregate_plotting/{results_filename}.sqlite")

db.execute("CREATE TABLE IF NOT EXISTS archived_files(path TEXT PRIMARY KEY, extension TEXT, content BLOB)")

for folder in COLLECT_FROM_FOLDERS:

    db.executemany("INSERT OR REPLACE INTO archived_files VALUES(?,?,?)", [(str(p), p.suffix, p.read_bytes()) for p in (root / folder).rglob("*") if p.is_file() and p.suffix in {".json", ".npz", ".txt"}])

db.commit()

df = pd.json_normalize([{"source_path": path, **json.loads(blob)} for path, blob in db.execute("SELECT path,content FROM archived_files WHERE extension='.json'") if isinstance(json.loads(blob), dict)])  # Raw pandas view; normalization, duplicate exclusion, and plotting require the full downloaded script.
print(db.execute("SELECT COUNT(*) FROM archived_files WHERE extension='.npz'").fetchone()[0], " npz files stored in the database.")

print("Exctracting .npz files into sql database")


db = sqlite3.connect(f"{root}/aggregate_plotting/{results_filename}.sqlite")
db.execute("CREATE TABLE IF NOT EXISTS npz_catalog(source_path TEXT,array_key TEXT,table_name TEXT,shape TEXT,dtype TEXT,PRIMARY KEY(source_path,array_key))")

for path, blob in db.execute("SELECT path,content FROM archived_files WHERE extension='.npz'"):
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        for key in data.files:
            a = data[key]; n = a.shape[0] if a.ndim else 1; width = int(np.prod(a.shape[1:])) if a.ndim else 1; rows = a.reshape(n, width); table = "npz_" + hashlib.sha256(key.encode()).hexdigest()[:16] + "_" + str(width); columns = [f"value_{i}" for i in range(width)]; db.execute(f"CREATE TABLE IF NOT EXISTS {table}(source_path TEXT,array_key TEXT,row_index INTEGER,{','.join(c + ' REAL' for c in columns)})"); db.execute(f"DELETE FROM {table} WHERE source_path=? AND array_key=?", (path,key)); db.execute("INSERT OR REPLACE INTO npz_catalog VALUES(?,?,?,?,?)", (path,key,table,json.dumps(a.shape),str(a.dtype))); db.commit()
            for start in range(0,n,1000): frame = pd.DataFrame(rows[start:start+1000],columns=columns,index=pd.RangeIndex(start,min(start+1000,n))); frame["source_path"] = path; frame["array_key"] = key; frame.to_sql(table,db,if_exists="append",index=True,index_label="row_index"); print(f"{path} | {key}: {min(start+1000,n)}/{n} rows",flush=True)
            print(f"Saved {key}, shape={a.shape}, table={table}",flush=True); del rows, a


