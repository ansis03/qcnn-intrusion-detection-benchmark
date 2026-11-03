"""Robustness checks and the classical baseline (paper Sec. V).

    python -m experiments.checks rescore  --dataset NSL-KDD   # thresholds from training-pool benign records / 0.1% target
    python -m experiments.checks long     --dataset NSL-KDD   # 1,000 training iterations at N=500
    python -m experiments.checks cnn      --dataset NSL-KDD   # one-layer CNN on the CB, MC and TB subsets
    python -m experiments.checks fullpool --dataset NSL-KDD   # the same CNN on the entire training pool

Results go to results/checks/<check>/<dataset>/. The 1,024-shot check needs no new
runs; analysis/checks.py resamples the released scores. rescore needs the final
parameters in results/main/<dataset>/models.npz. Circuits 1 and 9 (U_TTN, U_SU4) are
used for the QCNN checks.
"""
import argparse, json, time
import numpy as np
from src.data import ROOT, DATASETS, load, evaluation
from src.sampling import all_subsets
from src.model import initial_params, batch_positions, train, scores, evaluate, threshold

NS, SEEDS, CHECK_CIRCUITS, PAIRED = [50, 100, 500, 1000, 5000], list(range(42, 47)), ['U_TTN', 'U_SU4'], ['balance', 'coverage', 'uniform']
OUT = ROOT/'results/checks'


def rescore(ds, x, xt, y, yt, types, tt, ev):
    """Rescore the main-experiment models on more benign records, without retraining.

    test0.001 / test0.01: thresholds at 0.1% and 1% from half of a larger test-side benign pool
    (NSL-KDD: all 9,711 test benign records; CICIDS2017: 20,000), FPR on the other half.
    train0.01: 1% threshold from 10,000 training-pool benign records, FPR on the whole test-side pool.
    Attack scores are those of the same evaluation attacks.
    """
    rng = np.random.default_rng(20260928)
    tb = np.flatnonzero(yt == 0)
    if ds == 'CICIDS2017':
        tb = rng.choice(tb, 20000, replace=False)
    tb = rng.permutation(tb); cal, test = tb[:len(tb)//2], tb[len(tb)//2:]
    trb = rng.choice(np.flatnonzero(y == 0), 10000, replace=False)
    z = np.load(ROOT/'results/main'/ds/'models.npz')
    params = dict(zip(z['keys'].tolist(), z['final_params'])); main = dict(zip(z['keys'].tolist(), z['scores']))
    out = OUT/'rescore'/ds; out.mkdir(parents=True, exist_ok=True)
    for n in NS:
        for seed in SEEDS:
            for c in CHECK_CIRCUITS:
                for m in PAIRED:
                    key = f'{c}_N{n}_seed{seed}_{m}'
                    if (out/f'{key}.json').exists():
                        continue
                    p = params[key][~np.isnan(params[key])]; sc = main[key]
                    s_cal, s_test, s_tr = scores(xt[cal], p, c), scores(xt[test], p, c), scores(x[trb], p, c)
                    r = {'key': key, 'dataset': ds, 'N': n, 'seed': seed, 'circuit': c, 'method': m,
                         'n_cal': int(len(cal)), 'n_test_benign': int(len(test)), 'n_train_benign': int(len(trb))}
                    for tag, t, fpr_scores in [(f'test{a}', threshold(s_cal, a), s_test) for a in (0.001, 0.01)] + \
                                              [('train0.01', threshold(s_tr, .01), np.r_[s_cal, s_test])]:
                        rec = np.array([np.mean(sc[ev['types'] == k] > t) for k in ev['names']])
                        r[tag] = {'recall_macro': float(rec.mean()), 'recall_weighted': float(ev['weights'] @ rec),
                                  'fpr': float(np.mean(fpr_scores > t))}
                    (out/f'{key}.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
                    print(key, flush=True)


def long(ds, x, xt, y, yt, types, tt, ev, n=500, iterations=1000):
    """The N=500 QCNNs retrained with 1,000 iterations; everything else as in the main experiment."""
    out = OUT/'long'/ds; out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        sub = all_subsets(y, types, n, seed)
        positions = batch_positions(n, seed, iterations, base=300000)
        for c in CHECK_CIRCUITS:
            for m in PAIRED:
                key = f'{c}_N{n}_seed{seed}_{m}'
                if (out/f'{key}.json').exists():
                    continue
                tic = time.time()
                p, _ = train(x, y, sub[m], positions, initial_params(c, seed), c)
                r = {'key': key, 'dataset': ds, 'N': n, 'seed': seed, 'circuit': c, 'method': m, 'iterations': iterations,
                     **evaluate(scores(ev['x'], p, c), ev, types, sub[m])}
                (out/f'{key}.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
                print(key, round(r['recall_weighted'], 3), round(time.time() - tic), flush=True)


def cnn_train(x, y, init_seed, order_seed):
    """One-layer 1D CNN (src/classical.py): Adam 1e-3, 60 epochs, batch 32, BCE with logits."""
    import torch, torch.nn as nn
    from src.classical import CNN1D
    torch.manual_seed(init_seed)
    model = CNN1D(input_dim=8, n_filters=8)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.BCEWithLogitsLoss()
    X, Y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    g = torch.Generator().manual_seed(order_seed)
    model.train()
    for _ in range(60):
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 32):
            b = perm[i:i + 32]
            opt.zero_grad(); loss = loss_fn(model(X[b]), Y[b]); loss.backward(); opt.step()
    model.eval()
    return model


def cnn_scores(model, x):
    import torch
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).numpy().astype(float)


def cnn(ds, x, xt, y, yt, types, tt, ev, inits=range(5)):
    """The CNN trained on the same CB, MC and TB subsets; five initializations replace the circuits."""
    import torch
    torch.set_num_threads(1)
    out = OUT/'cnn'/ds; out.mkdir(parents=True, exist_ok=True)
    for n in NS:
        for seed in SEEDS:
            sub = all_subsets(y, types, n, seed)
            for init in inits:
                for m in PAIRED:
                    key = f'CNN{init}_N{n}_seed{seed}_{m}'
                    if (out/f'{key}.json').exists():
                        continue
                    idx = sub[m]
                    model = cnn_train(x[idx], y[idx], 500000 + 10*seed + init, 600000 + 10*seed + init + 1000*n)
                    r = {'key': key, 'dataset': ds, 'N': n, 'seed': seed, 'circuit': f'CNN{init}', 'method': m,
                         **evaluate(cnn_scores(model, ev['x']), ev, types, idx)}
                    (out/f'{key}.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
                    print(key, round(r['recall_weighted'], 3), flush=True)


def fullpool(ds, x, xt, y, yt, types, tt, ev, inits=range(5)):
    """The CNN trained on the entire training pool."""
    import torch
    torch.set_num_threads(1)
    out = OUT/'fullpool'/ds; out.mkdir(parents=True, exist_ok=True)
    for init in inits:
        key = f'CNNfull{init}'
        if (out/f'{key}.json').exists():
            continue
        tic = time.time()
        model = cnn_train(x, y, 700000 + init, 800000 + init)
        r = {'key': key, 'dataset': ds, 'N': int(len(x)), 'init': init, 'method': 'full_pool',
             **evaluate(cnn_scores(model, ev['x']), ev, types, np.arange(len(x))), 'seconds': time.time() - tic}
        (out/f'{key}.json').write_text(json.dumps(r, indent=1), encoding='utf-8')
        print(key, round(r['recall_weighted'], 3), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('check', choices=['rescore', 'long', 'cnn', 'fullpool'])
    ap.add_argument('--dataset', required=True, choices=DATASETS)
    args = ap.parse_args()
    x, xt, y, yt, types, tt = load(args.dataset)
    ev = evaluation(xt, yt, tt, y, types)
    globals()[args.check](args.dataset, x, xt, y, yt, types, tt, ev)


if __name__ == '__main__':
    main()
