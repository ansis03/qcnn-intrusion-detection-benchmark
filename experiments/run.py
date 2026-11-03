"""Main experiment: four sampling policies x five circuits x five subset sizes x five seeds per dataset.

    python -m experiments.run --dataset NSL-KDD                # all 500 models of one dataset
    python -m experiments.run --dataset NSL-KDD --seed 42      # one seed (run seeds in parallel)
    python -m experiments.run --dataset NSL-KDD --n 50 --circuits U_TTN --policies balance uniform

Writes one JSON summary per model to results/main/<dataset>/ (or --out) and, with
--save-scores, the evaluation scores and final parameters next to it. Models whose
JSON already exists are skipped, so an interrupted run can be resumed.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from src.data import ROOT, DATASETS, load, evaluation
from src.sampling import POLICIES, all_subsets
from src.model import CIRCUITS, initial_params, batch_positions, train, scores, evaluate

NS, SEEDS = [50, 100, 500, 1000, 5000], list(range(42, 47))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=DATASETS)
    ap.add_argument('--seed', type=int, nargs='*', default=SEEDS)
    ap.add_argument('--n', type=int, nargs='*', default=NS)
    ap.add_argument('--circuits', nargs='*', default=list(CIRCUITS), choices=list(CIRCUITS))
    ap.add_argument('--policies', nargs='*', default=POLICIES, choices=POLICIES)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--save-scores', action='store_true')
    args = ap.parse_args()
    ds = args.dataset
    out = args.out or ROOT/'results/main'/ds
    out.mkdir(parents=True, exist_ok=True)
    x, xt, y, yt, types, tt = load(ds)
    ev = evaluation(xt, yt, tt, y, types)
    print(f'{ds}: pool {len(y)} ({y.mean():.3f} attacks), evaluation {len(ev["labels"])} records, '
          f'{len(ev["names"])} attack types', flush=True)
    for n in args.n:
        for seed in args.seed:
            subsets = all_subsets(y, types, n, seed)
            positions = batch_positions(n, seed)
            for name in args.circuits:
                init = initial_params(name, seed)
                for policy in args.policies:
                    key = f'{name}_N{n}_seed{seed}_{policy}'
                    if (out/f'{key}.json').exists():
                        continue
                    tic = time.time(); idx = subsets[policy]
                    params, losses = train(x, y, idx, positions, init, name)
                    s = scores(ev['x'], params, name)
                    assert np.isfinite(s).all() and np.isfinite(losses).all()
                    r = {'key': key, 'dataset': ds, 'N': n, 'seed': seed, 'circuit': name, 'method': policy,
                         'attack_records': int(y[idx].sum()), 'selected_types': len(set(types[idx][y[idx] == 1])),
                         **evaluate(s, ev, types, idx)}
                    (out/f'{key}.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
                    if args.save_scores:
                        np.savez_compressed(out/f'{key}.npz', scores=s, final_params=params, losses=losses,
                                            selected_train_rows=idx)
                    print(f'{key}: AUC {r["auc_macro"]:.3f}, recall {r["recall_weighted"]:.3f}, '
                          f'{time.time() - tic:.0f}s', flush=True)


if __name__ == '__main__':
    main()
