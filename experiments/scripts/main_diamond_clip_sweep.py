#!/usr/bin/env python3
"""

"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

THRESHOLDS = (.1, .01, .001)


def configurations():
    result = []
    for seed in range(3):
        block = [dict(method=m, lr=lr, clip=clip, seed=seed)
                 for m, lr, clip in itertools.product(
                     ['ld', 'basis_kw0.1'], [3e-4, 3e-3], [.1, 1., None])]
        random.Random(5401 + seed).shuffle(block)
        result.extend(block)
    return result


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def peak_rss_mib():
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 2 if sys.platform == 'darwin' else 1024)


def worker(config_path):
    start = time.monotonic()  # includes dependency imports and initialization
    config = json.loads(Path(config_path).read_text())
    result_path = Path(config['result_path'])
    result = dict(config=config, status='starting', curve=[], updates=0,
                  clipped_steps=0, grad_norm_sum=0., train_seconds=0.)
    def save():
        result['elapsed_seconds'] = time.monotonic() - start
        result['peak_rss_mib'] = peak_rss_mib()
        atomic_json(result_path, result)
    try:
        os.environ['OMP_NUM_THREADS'] = str(config['threads'])
        os.environ['MKL_NUM_THREADS'] = str(config['threads'])
        sys.path.insert(0, str(Path.cwd()))
        import torch
        import numpy as np
        import theory.kc_exact as k
        from experiments.scripts.loop_gfn import FreeOrderPolicy
        torch.set_num_threads(config['threads'])
        # Importing the provided exact engine sets float64 globally; make it explicit.
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        g = k.ExactDAG(9, 2.27, J_seed=0, dag='free', kind='ising')
        pol = FreeOrderPolicy(9, hidden=128)
        opt = torch.optim.Adam(pol.parameters(), lr=config['lr'])
        states = torch.arange(3 ** 9)
        digits = (states[:, None] // (3 ** torch.arange(9))) % 3
        spins = (digits == 2).double() - (digits == 1).double()
        mask = (digits > 0).double()
        which = k.which_of(config['method'])
        expected = (1., 0., 0., 0.) if config['method'] == 'ld' else (0., .1, 1., 0.)
        if tuple(k.method_weights(config['method'])) != expected:
            raise RuntimeError('Installed method weights differ from the supplied source')
        result['resolved'] = dict(D=9, T=2.27, kind='ising', dag='free', grad='exact',
            weights='onpolicy', loss_weights=expected, policy='FreeOrderPolicy(hidden=128)',
            schedule='constant', dtype='float64', torch_version=str(torch.__version__),
            adam_betas=[.9, .999], adam_eps=1e-8, threads=torch.get_num_threads())
        result['source_sha256'] = {}
        for source in [Path(k.__file__), Path('loop_gfn.py'), Path('spin_glass_lib.py'), Path(__file__)]:
            if source.is_file():
                result['source_sha256'][str(source.resolve())] = hashlib.sha256(source.read_bytes()).hexdigest()
        def evaluate():
            with torch.no_grad():
                th = pol(spins, mask, g.level)
                metrics = g.metrics(th)
                losses = {name: float(value) for name, value in g.losses(th, which, 'onpolicy').items()}
                values = dict(metrics, **losses)
                if not all(math.isfinite(float(x)) for x in values.values()):
                    raise FloatingPointError('Nonfinite evaluation')
            record = dict(it=result['updates'], train_seconds=result['train_seconds'],
                          elapsed_seconds=time.monotonic() - start,
                          clip_fraction=result['clipped_steps']/max(1, result['updates']),
                          mean_preclip_grad_norm=result['grad_norm_sum']/max(1, result['updates']), **values)
            result['curve'].append(record)
            result['final'] = record
            save()
        result['status'] = 'running'
        evaluate()
        # Reserve time for final evaluation and writing inside the parent's larger hard cap.
        training_deadline = start + config['soft_seconds']
        last_eval = time.monotonic()
        while result['updates'] < config['max_iters'] and time.monotonic() < training_deadline:
            step_start = time.monotonic()
            th = pol(spins, mask, g.level)
            terms = g.losses(th, which, 'onpolicy')
            loss = k.objective(terms, config['method'])
            opt.zero_grad()
            loss.backward()
            # Infinity obtains the same preclip norm without applying a finite clipping threshold.
            bound = float('inf') if config['clip'] is None else config['clip']
            norm = float(torch.nn.utils.clip_grad_norm_(pol.parameters(), bound))
            if not math.isfinite(norm) or not math.isfinite(float(loss.detach())):
                raise FloatingPointError('Nonfinite loss or gradient')
            opt.step()
            result['train_seconds'] += time.monotonic() - step_start
            result['updates'] += 1
            result['grad_norm_sum'] += norm
            result['clipped_steps'] += int(config['clip'] is not None and norm > config['clip'])
            if result['updates'] == 1 or time.monotonic() - last_eval >= 5.:
                evaluate()
                last_eval = time.monotonic()
        if result['curve'][-1]['it'] != result['updates']:
            evaluate()
        result['status'] = 'iteration_cap' if result['updates'] >= config['max_iters'] else 'time_cap'
        save()
        return 0
    except Exception:
        result['status'] = 'error'
        result['error'] = traceback.format_exc()
        save()
        print(result['error'], file=sys.stderr)
        return 1


def summarize(directory):
    fields = ['run', 'method', 'lr', 'clip', 'seed', 'status', 'updates', 'final_kl_qp',
              'final_kl_pq', 'final_tv', 'best_kl_qp', 'train_seconds', 'elapsed_seconds',
              'peak_rss_mib', 'clip_fraction', 'mean_preclip_grad_norm', 'slot_wall_seconds']
    for threshold in THRESHOLDS:
        fields += [f'first_kl_le_{threshold}_train_s', f'first_kl_le_{threshold}_wall_s']
    rows = []
    for path in sorted(Path(directory).glob('run_*.json')):
        r = json.loads(path.read_text())
        if 'config' not in r:
            continue
        c, final, curve = r['config'], r.get('final', {}), r.get('curve', [])
        row = dict(run=path.stem, method=c['method'], lr=c['lr'],
                   clip='off' if c['clip'] is None else c['clip'], seed=c['seed'],
                   status=r['status'], updates=r.get('updates', 0),
                   final_kl_qp=final.get('kl_qp'), final_kl_pq=final.get('kl_pq'), final_tv=final.get('tv'),
                   best_kl_qp=min((p['kl_qp'] for p in curve), default=None),
                   train_seconds=r.get('train_seconds'), elapsed_seconds=r.get('elapsed_seconds'),
                   peak_rss_mib=r.get('peak_rss_mib'), clip_fraction=final.get('clip_fraction'),
                   mean_preclip_grad_norm=final.get('mean_preclip_grad_norm'), slot_wall_seconds=r.get('slot_wall_seconds'))
        for threshold in THRESHOLDS:
            crossings = [p for p in curve if p['kl_qp'] <= threshold]
            row[f'first_kl_le_{threshold}_train_s'] = crossings[0]['train_seconds'] if crossings else None
            row[f'first_kl_le_{threshold}_wall_s'] = crossings[0]['elapsed_seconds'] if crossings else None
        rows.append(row)
    with (Path(directory) / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


NOTES = '''INTERPRETATION
This screen changes clipping threshold and learning rate, keeping the local loss SD+.1*kite
fixed. The supplied runs document LR/weight sweeps and norm-1 clipping, but not this
clipping-by-objective screen. This is new relative to those records, not a claim of literature novelty.
Question: does changing clipping help local and global objectives differently?
Compare clip=off/.1/1 WITHIN each method and LR, paired by seed. Then compare the change
between methods. Select by median final KL or time to a prespecified KL threshold, not best seed.
A local-specific benefit would motivate a fresh-seed confirmation; no benefit is also informative.
Do not assume clipping is bad, or promise that either method will win.

Each worker gets the same wall-time slot (including imports/setup/evaluation), so update counts
can differ. Curves separately record active optimizer time, elapsed time, KL in both directions,
TV, preclip gradient norms and clipped-step fractions. Diagnostics/initialization are excluded
from active optimizer time. Peak RSS includes Python/PyTorch/data, not just model tensors.
First-threshold crossings are approximate and sampled about every five seconds, not guaranteed
sustained convergence. Read whole curves; empty threshold fields mean not reached by budget.
The memory/seconds recorded here characterize this exact-enumeration implementation.
They do NOT establish that sampled local training is cheaper at D=100. That needs a sampled test.

Three seeds, two LRs and a short budget are exploratory. This does not exhaust the tuning space,
and it may not reach the KL values in longer historical runs. Constant LR intentionally avoids
stretching a cosine schedule across unequal update counts. Checkpoints are result curves only,
not optimizer-state checkpoints. Source modules are never overwritten.
Use the same source/dependencies/hardware for comparisons. Execution is CPU-only, two threads
by default, no entropy bonus or beta annealing. On-policy expectation weights stay detached.
After a pilot, confirm the selected settings against the SAME-LR norm-1 baseline on fresh seeds.
This script was syntax-checked and its configuration/summary logic checked here; numerical
training was not run here because PyTorch and loop_gfn.py were unavailable.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--minutes', type=float, default=110., help='whole sweep cap, maximum 115 minutes')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--outdir')
    parser.add_argument('--max-iters', type=int, default=8000)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.describe:
        print(json.dumps(configurations(), indent=2)); print(NOTES); return 0
    if not 0 < args.minutes <= 115 or args.threads < 1 or args.max_iters < 1:
        parser.error('Require 0<minutes<=115, threads>=1, max-iters>=1')
    for name in ['kc_exact.py', 'loop_gfn.py', 'spin_glass_lib.py']:
        if not Path(name).is_file():
            parser.error(f'Run in the research project root containing {name}')
    parent_start = time.monotonic()
    deadline = parent_start + args.minutes*60
    directory = Path(args.outdir or time.strftime('outputs_clip_lr_%Y%m%d_%H%M%S')).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'README.txt').write_text(NOTES)
    configs = configurations()
    # Reserve ten seconds for final summaries; no run is given more than three minutes.
    hard_slot = min(180., max(0., (args.minutes*60-10.)/len(configs)))
    print(f'{len(configs)} configurations; at most {hard_slot:.1f}s each; outputs: {directory}', flush=True)
    script = str(Path(__file__).resolve())
    for i, config in enumerate(configs):
        remaining = deadline-time.monotonic()-5.
        if remaining <= 1:
            break
        allowance = min(hard_slot, remaining)
        path = directory/f'run_{i:02d}.json'
        config.update(threads=args.threads, max_iters=args.max_iters,
                      soft_seconds=max(.1, allowance-15.), result_path=str(path))
        config_path = directory/f'config_{i:02d}.json'
        atomic_json(config_path, config)
        log = directory/f'run_{i:02d}.log'
        print(f'[{i+1}/36] {config["method"]}, lr={config["lr"]}, clip={config["clip"]}, seed={config["seed"]}', flush=True)
        slot_start = time.monotonic()
        with log.open('w') as output:
            process = subprocess.Popen([sys.executable, script, '--worker', str(config_path)],
                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            killed = False
            try:
                status = process.wait(timeout=allowance)
            except subprocess.TimeoutExpired:
                killed = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                status = process.returncode
        if path.exists():
            record = json.loads(path.read_text())
        else:
            record = dict(config=config, curve=[], updates=0)
        if killed:
            record['status'] = 'hard_time_cap'
        elif status and record.get('status') != 'error':
            record['status'] = 'worker_failed'
        record['slot_wall_seconds'] = time.monotonic() - slot_start
        atomic_json(path, record)
        summarize(directory)
        if status and not killed:
            print(f'Worker failed; inspect {log}. Stopping instead of repeating the failure.', file=sys.stderr)
            return 1
    rows = summarize(directory)
    print(f'Finished {len(rows)} slots in {(time.monotonic()-parent_start)/60:.1f} minutes.', flush=True)
    print('Results:',directory/'summary.csv',flush=True)
    print('This is a short-budget pilot; compare full curves and confirm any finding on fresh seeds.',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
