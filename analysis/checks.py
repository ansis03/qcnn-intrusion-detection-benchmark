"""Summaries of the robustness checks and the classical baseline (paper Sec. V), from results/checks/.

    python -m analysis.checks

For NSL-KDD and CICIDS2017 with circuits 1 and 9, the change TB-CB in Rec_w is tested
(paired Wilcoxon over two circuits x five seeds) at every N under four variations:
thresholds from 10,000 training-pool benign records, a 0.1% target FPR, evaluation
scores resampled from 1,024 shots, and (N=500 only) 1,000 training iterations.
Writes analysis/checks.json.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from analysis.load import ROOT, DS, NS, runs, scores, evaluation

C = ROOT/'results/checks'
read = lambda check, ds: [json.loads(q.read_text(encoding='utf-8')) for q in sorted((C/check/ds).glob('*.json'))]
out = {'robustness': {}, 'cnn': {}}


def tb_minus_cb(df, k='recall_weighted'):
    w = df.pivot_table(index=['circuit', 'seed'], columns='method', values=k)
    x = (w['uniform'] - w['balance']).values
    return {'mean': float(x.mean()), 'negative': int((x < 0).sum()), 'n': int(len(x)), 'p': float(wilcoxon(x).pvalue)}


SHOTS = 1024
for ds in DS[:2]:
    res = {}
    # thresholds from training-pool benign records, and a 0.1% target
    rs = pd.DataFrame([{'N': r['N'], 'seed': r['seed'], 'circuit': r['circuit'], 'method': r['method'],
                        **{f'{t}|{k}': r[t][k] for t in ('test0.001', 'test0.01', 'train0.01') for k in ('recall_weighted', 'fpr')}}
                       for r in read('rescore', ds)])
    for tag, label in (('train0.01', 'training-pool threshold'), ('test0.001', '0.1% target')):
        res[label] = {n: tb_minus_cb(rs[rs.N == n].rename(columns={f'{tag}|recall_weighted': 'recall_weighted'})) for n in NS}
    # evaluation scores resampled from 1,024 shots: <Z> = s becomes 2k/S - 1 with k ~ Binomial(S, (1 + s)/2)
    ev, S, R = evaluation(ds), scores(ds), runs(ds)
    names = sorted(set(ev['types'][ev['labels'] == 1]))
    shot = []
    for _, r in R[R.circuit.isin(['U_TTN', 'U_SU4']) & R.method.isin(['balance', 'uniform'])].iterrows():
        rng = np.random.default_rng(900000 + 10*r.seed + (r.method == 'uniform'))
        s = 2*rng.binomial(SHOTS, np.clip((1 + S[r.key])/2, 0, 1))/SHOTS - 1
        thr = np.sort(s[ev['calibration_positions']])[1484]
        rec = np.array([np.mean(s[ev['types'] == k] > thr) for k in names])
        shot.append({'N': r.N, 'seed': r.seed, 'circuit': r.circuit, 'method': r.method, 'recall_weighted': float(ev['weights'] @ rec)})
    shot = pd.DataFrame(shot)
    res['1,024 shots'] = {n: tb_minus_cb(shot[shot.N == n]) for n in NS}
    # 1,000 training iterations at N=500
    res['1,000 iterations'] = {500: tb_minus_cb(pd.DataFrame(read('long', ds)))}
    out['robustness'][ds] = res
pmax = max(c['p'] for ds in out['robustness'] for v in out['robustness'][ds].values() for c in v.values())
allneg = all(c['mean'] < 0 for ds in out['robustness'] for v in out['robustness'][ds].values() for c in v.values())
out['robustness']['max_p'] = pmax
print(f'robustness: TB-CB in Rec_w negative in every setting: {allneg}; largest p = {pmax:.4f}')

# classical CNN: TB-CB on the same subsets, and the CNN trained on the entire pool
for ds in DS:
    cn = pd.DataFrame(read('cnn', ds))
    full = float(np.median([r['recall_weighted'] for r in read('fullpool', ds)]))
    cb5k = float(cn[(cn.N == 5000) & (cn.method == 'balance')].recall_weighted.median())
    tb = {}
    if ds != 'TON_IoT':
        for n in NS:
            w = cn[cn.N == n].pivot_table(index=['circuit', 'seed'], columns='method', values='recall_weighted')
            x = (w['uniform'] - w['balance']).values
            tb[n] = {'median': float(np.median(x)), 'p': float(wilcoxon(x).pvalue)}
    out['cnn'][ds] = {'fullpool_median_Rw': full, 'cb5000_median_Rw': cb5k, 'tb_minus_cb': tb}
    print(f'CNN {ds}: full pool {full:.3f} vs CB N=5,000 {cb5k:.3f}',
          '| TB-CB median ' + ', '.join(f'{n}: {v["median"]:+.3f} (p={v["p"]:.3f})' for n, v in tb.items()))
(ROOT/'analysis/checks.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
