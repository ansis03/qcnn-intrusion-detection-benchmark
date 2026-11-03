"""Tables II-IV, Figs. 1-3 and the numbers cited in the paper, from results/main/.

    python -m analysis.make_assets

Writes tables/{overall,bysize,ansatz4}.tex (table bodies as in the paper),
figures/{fig_overall,fig_scores,fig_types,fig_sweep}.{pdf,png} and analysis/numbers.json.
Summary statistic: median (Q1-Q3) over models; paired Wilcoxon signed-rank tests against CB.
"""
import json
import numpy as np
from scipy.stats import wilcoxon, norm, gaussian_kde, binomtest, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from analysis.load import ROOT, DS, POLICIES, SHORT, NAME, CIRCUITS, NS, runs, per_type, scores, evaluation, pool_shares

F = ROOT/'figures'; T = ROOT/'tables'
F.mkdir(exist_ok=True); T.mkdir(exist_ok=True)
TEXDS = {'NSL-KDD': 'NSL-KDD', 'CICIDS2017': 'CICIDS2017', 'TON_IoT': r'\ton'}
COL = {'stratified': '.5', 'balance': '#0072B2', 'coverage': '#009E73', 'uniform': '#D55E00'}
MRK = {'stratified': 'D', 'balance': 'o', 'coverage': 's', 'uniform': '^'}
GCOL = {'benign': '.45', 'freq': '#D55E00', 'rare': '#0072B2'}
GRID = [.001, .002, .005, .01, .02, .03, .05, .07, .10, .15, .20, .30]
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.axisbelow': True})
R = {d: runs(d) for d in DS}
num = {'overall': {}, 'bysize': {}, 'paired': {}, 'ansatz': {}, 'groups': {}, 'scores': {}, 'types': {}, 'sweep': {}, 'misc': {}}
f2 = lambda v: f'{v:.2f}'
q3 = lambda v: [float(x) for x in np.percentile(v, [25, 50, 75])]


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(F/f'{name}.{ext}', dpi=230, bbox_inches='tight', pad_inches=.04)
    plt.close(fig)


def rows(name, lines):
    (T/name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def paired(d, sel, a, b, k):
    w = R[d][sel(R[d])].pivot_table(index=['N', 'circuit', 'seed'], columns='method', values=k)
    x = (w[a] - w[b]).values
    p = float(wilcoxon(x).pvalue) if np.any(x != 0) else 1.0
    return {'median': float(np.median(x)), 'mean': float(x.mean()), 'neg': int((x < 0).sum()), 'pos': int((x > 0).sum()),
            'n': len(x), 'p': p}


# ---------------------------------------------------------------- Table II and Fig. 1: overall
for d in DS:
    num['overall'][d] = {p: {k: q3(R[d][R[d].method == p][k]) for k in ('auc_macro', 'auc_weighted', 'recall_macro', 'recall_weighted', 'fpr')}
                         for p in POLICIES}
lines = []
for p in POLICIES:
    cells = []
    for d in DS:
        o = num['overall'][d][p]
        cells += [f2(o['auc_macro'][1]), f2(o['auc_weighted'][1]), f2(o['recall_macro'][1]),
                  f"{f2(o['recall_weighted'][1])} ({f2(o['recall_weighted'][0])}--{f2(o['recall_weighted'][2])})"]
    lines.append(f'{NAME[p]} & ' + ' & '.join(cells) + r' \\')
rows('overall.tex', lines)

fig, axes = plt.subplots(2, 1, figsize=(3.45, 2.75), sharex=True)
for ax, (k, lab) in zip(axes, [('auc_macro', r'$\overline{\mathrm{AUC}}$'), ('recall_weighted', r'$\mathrm{Rec}_w$ at 1% FPR')]):
    w = .8/len(POLICIES)
    for i, p in enumerate(POLICIES):
        st = np.array([num['overall'][d][p][k] for d in DS])
        x = np.arange(3) + (i - (len(POLICIES) - 1)/2)*w
        ax.bar(x, st[:, 1], w*.9, color=COL[p], label=NAME[p], alpha=.88)
        ax.errorbar(x, st[:, 1], yerr=[st[:, 1] - st[:, 0], st[:, 2] - st[:, 1]], fmt='none', ecolor='k', lw=.6, capsize=1.3)
    ax.set_xticks(range(3), DS, fontsize=7); ax.set_ylabel(lab, fontsize=7.5); ax.grid(axis='y', alpha=.25)
    ax.tick_params(labelsize=7); ax.set_ylim(.5 if k == 'auc_macro' else 0, 1)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.54, 1.0), ncol=2, frameon=False,
           fontsize=7, handlelength=1.2, columnspacing=1.0)
fig.subplots_adjust(left=.15, right=.98, top=.84, bottom=.08, hspace=.12)
fig.align_ylabels(axes)
save(fig, 'fig_overall')

# ---------------------------------------------------------------- Table III: by subset size, paired tests
lines = []
for d in DS:
    num['bysize'][d] = {}; num['paired'][d] = {}
    for i, n in enumerate(NS):
        sel = lambda r, n=n: r.N == n
        num['bysize'][d][n] = {p: {k: q3(R[d][(R[d].N == n) & (R[d].method == p)][k]) for k in ('recall_weighted', 'auc_macro', 'recall_macro', 'fpr')}
                               for p in POLICIES}
        num['paired'][d][n] = {f'{SHORT[a]}-{SHORT[b]}|{k}': paired(d, sel, a, b, k)
                               for a, b in (('stratified', 'balance'), ('coverage', 'balance'), ('uniform', 'balance'),
                                            ('uniform', 'coverage'), ('uniform', 'stratified'))
                               for k in ('recall_weighted', 'auc_macro', 'recall_macro')}
        cells = []
        for k, iqr in (('recall_weighted', True), ('auc_macro', False)):
            for p in POLICIES:
                lo, me, hi = num['bysize'][d][n][p][k]
                star = '' if p == 'balance' else ('$^{*}$' if num['paired'][d][n][f'{SHORT[p]}-CB|{k}']['p'] < .05 else '')
                cells.append(f'{f2(me)}{star} ({f2(lo)}--{f2(hi)})' if iqr else f'{f2(me)}{star}')
        lab = (r'\multirow{5}{*}{' + TEXDS[d] + '}') if i == 0 else ''
        lines.append(f'{lab} & {n:,} & ' + ' & '.join(cells) + r' \\')
    if d != DS[-1]:
        lines.append(r'\midrule')
rows('bysize.tex', lines)

# Holm over the ten TB-CB tests in Rec_w, and a sign test over circuit-averaged subsets (the independent unit)
ps = sorted((num['paired'][d][n]['TB-CB|recall_weighted']['p'], d, n) for d in DS[:2] for n in NS)
num['misc']['holm_max'] = float(max(min(1, p*(10 - i)) for i, (p, _, _) in enumerate(ps)))
for d in DS[:2]:
    w = R[d].pivot_table(index=['N', 'seed', 'circuit'], columns='method', values='recall_weighted')
    sub = (w['uniform'] - w['balance']).groupby(level=[0, 1]).mean()
    num['misc'][f'sign_{d}'] = {'neg': int((sub < 0).sum()), 'n': len(sub), 'p': float(binomtest(int((sub < 0).sum()), len(sub)).pvalue)}

# ---------------------------------------------------------------- Table IV: by circuit
lines = []
for c, (cn, par) in CIRCUITS.items():
    cells = []
    for d in DS:
        for k in ('recall_weighted', 'auc_macro'):
            for p in POLICIES:
                v = float(R[d][(R[d].circuit == c) & (R[d].method == p)][k].median())
                num['ansatz'].setdefault(d, {}).setdefault(c, {}).setdefault(p, {})[k] = v
                cells.append(f2(v))
    lines.append(f'{cn} ({par}) & ' + ' & '.join(cells) + r' \\')
rows('ansatz4.tex', lines)
for d in DS[:2]:
    loss, neg = {}, 0
    for c in CIRCUITS:
        w = R[d][R[d].circuit == c].pivot_table(index=['N', 'seed'], columns='method', values='recall_weighted')
        loss[c] = float(np.median(w['uniform'] - w['balance'])); neg += int(((w['uniform'] - w['balance']) < 0).sum())
    rho = spearmanr([CIRCUITS[c][1] for c in CIRCUITS], [-loss[c] for c in CIRCUITS]).statistic
    num['misc'][f'circuit_loss_{d}'] = {'loss': loss, 'rho': float(rho)}
    num['misc'][f'circuit_neg_{d}'] = neg

# ---------------------------------------------------------------- Fig. 2: score distributions; recall against target FPR
M = 1500; ZT = norm.ppf(.99); TOP = norm.ppf(1 - .5/M); xs = np.linspace(-3.6, 3.3, 500)


def zcal(v, cs):
    """Score as a normal quantile of its rank among the 1,500 calibration benign scores."""
    p = (np.searchsorted(cs, v, 'left') + np.searchsorted(cs, v, 'right'))/2/M
    return norm.ppf(np.clip(p, .5/M, 1 - .5/M))


share = {d: pool_shares(d) for d in DS}
fig, axes = plt.subplots(4, 3, figsize=(7.05, 3.15), sharex=True, sharey=True)
sweep = {}
for j, d in enumerate(DS):
    ev = evaluation(d); S = scores(d)
    lab, ty, cal, testn = ev['labels'], ev['types'], ev['calibration_positions'], ev['test_normal_positions']
    names = sorted(set(ty[lab == 1]))
    freq = [k for k in names if share[d].get(k, 0) >= .10]                              # at least 10% of training attacks
    rare = [k for k in names if share[d].get(k, 1) < .01 and np.sum(ty == k) >= 30]      # below 1%, at least 30 evaluation attacks
    num['groups'][d] = {'frequent': freq, 'rare': rare}
    idx = {'benign': testn, 'freq': np.flatnonzero(np.isin(ty, freq)), 'rare': np.flatnonzero(np.isin(ty, rare))}
    att = {k: np.flatnonzero(ty == k) for k in names}
    num['scores'][d] = {}; sweep[d] = {}
    for i, p in enumerate(POLICIES):
        pooled = {g: [] for g in idx}; hit = {g: [] for g in idx}; curves = []
        for key in R[d][R[d].method == p].key:
            s = S[key]; cs = np.sort(s[cal]); thr = cs[1484]
            for g, ii in idx.items():
                pooled[g].append(zcal(s[ii], cs)); hit[g].append(s[ii] > thr)
            curves.append([np.mean([np.mean(s[att[k]] > cs[M - int(np.floor(a*M + 1e-9)) - 1]) for k in names]) for a in GRID])
        sweep[d][p] = np.array(curves)
        ax = axes[i, j]; rec = {}
        for g, ls, labl in (('benign', '-', 'Benign (test)'), ('freq', '-', 'Frequent attack types'), ('rare', '--', 'Rare attack types')):
            v = np.concatenate(pooled[g]); inside = v[v < TOP - 1e-9]
            yk = gaussian_kde(inside, bw_method=.18)(xs)*len(inside)/len(v)
            rec[g] = float(np.mean(np.concatenate(hit[g])))
            ax.plot(xs, yk, color=GCOL[g], ls=ls, lw=1.1, label=labl); ax.fill_between(xs, yk, color=GCOL[g], alpha=.08, lw=0)
        num['scores'][d][p] = rec
        ax.axvline(ZT, color='k', ls=':', lw=.8)
        ax.text(.03, .93, f"frequent {rec['freq']:.2f}\nrare {rec['rare']:.2f}", transform=ax.transAxes, ha='left', va='top',
                fontsize=6.3, linespacing=1.1)
        ax.tick_params(labelsize=6.8); ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(0, 1.2); ax.set_yticks([0, .5, 1])
        if i == 0:
            ax.set_title(d, fontsize=8.5)
        if j == 0:
            ax.set_ylabel(NAME[p].replace(' ', '\n').replace('-', '-\n'), fontsize=7, linespacing=1.0)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.53, -.01), ncol=3, fontsize=7.5,
           frameon=False, handlelength=2.0, columnspacing=1.2)
fig.supxlabel('Score relative to the calibration benign records (z)', fontsize=7.5, y=.075)
fig.subplots_adjust(left=.075, right=.99, bottom=.2, top=.92, wspace=.05, hspace=.14)
fig.align_ylabels(axes[:, 0])
save(fig, 'fig_scores')

# ---------------------------------------------------------------- Fig. 3: per-type medians
fmt = lambda v: '1.0' if v >= .995 else f'{v:.2f}'[1:]
short = lambda t: t.replace('Web Attack – ', 'Web ').replace('Sql Injection', 'SQL Injection')
fig, axs = plt.subplots(1, 8, figsize=(7.05, 2.7), gridspec_kw={'width_ratios': [1, 1, 1.5, 1, 1, 1.05, 1, 1]})
for g in (axs[2], axs[5]):
    g.axis('off')
for j, d in enumerate(DS):
    P = per_type(d); ev = evaluation(d)
    cnt = {k: int(np.sum((ev['types'] == k) & (ev['labels'] == 1))) for k in P.type.unique()}
    order = sorted(P.type.unique(), key=lambda k: -share[d].get(k, 0))
    med = P.groupby(['type', 'method'])[['auc', 'recall']].median()
    sub = P[(P.N == 500)].groupby(['type', 'method']).subset_count.mean()
    num['types'][d] = {k: {p: {'auc': float(med.loc[(k, p), 'auc']), 'recall': float(med.loc[(k, p), 'recall']),
                               'subset_N500': float(sub.loc[(k, p)])} for p in POLICIES} for k in order}
    num['types'][d]['_share'] = {k: float(share[d].get(k, 0)) for k in order}
    num['types'][d]['_eval_count'] = cnt
    for m, (k, lab) in enumerate([('auc', 'AUC'), ('recall', 'Recall at 1% FPR')]):
        ax = axs[(0, 3, 6)[j] + m]
        mat = med[k].unstack()[POLICIES].loc[order].values
        ax.imshow(mat, vmin=.5 if k == 'auc' else 0, vmax=1, cmap='viridis', aspect='auto')
        for a in range(mat.shape[0]):
            for b in range(mat.shape[1]):
                ax.text(b, a, fmt(mat[a, b]), ha='center', va='center', fontsize=4.3,
                        color='w' if mat[a, b] < (.3 if k == 'recall' else .7) else 'k')
        ax.set_xticks(range(4), [SHORT[p] for p in POLICIES], fontsize=4.3)
        pct = lambda t: '<0.1%' if 100*share[d].get(t, 0) < .05 else f'{100*share[d].get(t, 0):.1f}%'
        star = lambda t: '*' if cnt[t] < 30 else ''   # fewer than 30 evaluation attacks: uncertain estimate
        ax.set_yticks(range(len(order)), [f'{short(t)}{star(t)} ({pct(t)})' for t in order] if m == 0 else [], fontsize=4.8)
        ax.set_title(lab, fontsize=5.6); ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
fig.subplots_adjust(left=.14, right=.995, top=.885, bottom=.045, wspace=.06)
for j, d in enumerate(DS):
    a0, a1 = axs[(0, 3, 6)[j]].get_position(), axs[(0, 3, 6)[j] + 1].get_position()
    fig.text((a0.x0 + a1.x1)/2, .955, d, ha='center', va='bottom', fontsize=6.8)
save(fig, 'fig_types')

# ---------------------------------------------------------------- recall against target FPR (not in the paper; cited in the text)
fig, axes = plt.subplots(1, 3, figsize=(3.45, 2.0), sharey=True)
for ax, d in zip(axes, DS):
    num['sweep'][d] = {}
    for k_, p in enumerate(POLICIES):
        med = np.median(sweep[d][p], axis=0)
        num['sweep'][d][p] = {f'{a:g}': float(v) for a, v in zip(GRID, med)}
        ax.plot(100*np.array(GRID)*np.exp((k_ - 1.5)*.05), med, color=COL[p], marker=MRK[p], ms=2.4, lw=1.0, label=NAME[p], zorder=k_ + 2)
    ax.axvline(1, color='.6', lw=.7, ls=':')
    ax.set_xscale('log'); ax.set_xticks([.1, 1, 10], ['0.1', '1', '10']); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(d, fontsize=8.5); ax.set_xlabel('Target FPR (%)', fontsize=8); ax.tick_params(labelsize=7.5)
    ax.grid(alpha=.2); ax.set_ylim(0, 1)
axes[0].set_ylabel(r'Type-averaged $\overline{\mathrm{Rec}}$', fontsize=8)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(.55, 1.02), ncol=2, fontsize=7,
           frameon=False, handlelength=1.4, columnspacing=.9)
fig.subplots_adjust(left=.15, right=.98, bottom=.17, top=.71, wspace=.1)
save(fig, 'fig_sweep')

(ROOT/'analysis/numbers.json').write_text(json.dumps(num, indent=1, ensure_ascii=False), encoding='utf-8')
print('wrote tables/, figures/ and analysis/numbers.json')
