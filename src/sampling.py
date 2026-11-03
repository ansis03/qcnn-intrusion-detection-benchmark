"""The four sampling policies (paper Sec. III-B), built in a paired design.

For a given dataset, subset size n and seed, all four subsets start from the same
class-balanced draw (CB). Minimum coverage (MC) and type balancing (TB) replace attack
records, stratified sampling (ST) replaces records with records of the other class, and
records common to two subsets keep their positions in the training order.

Policy names in code and result files: stratified = ST, balance = CB,
coverage = MC, uniform = TB.
"""
import numpy as np

POLICIES = ['stratified', 'balance', 'coverage', 'uniform']
SHORT = {'stratified': 'ST', 'balance': 'CB', 'coverage': 'MC', 'uniform': 'TB'}


def paired_subsets(y, types, n, seed):
    """CB, MC and TB subsets (row indices into the training pool) for one n and seed.

    CB: n/2 benign and n/2 attack records drawn uniformly.
    MC: CB with the smallest number of substitutions that includes every attack type;
        each missing type replaces a record of a type that has more than one record.
    TB: the attack half spread as evenly as possible across types (water-filling),
        keeping MC's records where possible.
    The benign records and the final shuffle are shared by all three.
    """
    rng = np.random.default_rng(seed)
    benign = rng.choice(np.flatnonzero(y == 0), n//2, replace=False)
    attacks = np.flatnonzero(y == 1)
    cb = rng.choice(attacks, n//2, replace=False)
    names = list(dict.fromkeys(types[attacks].tolist()))   # order of first occurrence in the pool
    pools = {k: rng.permutation(attacks[types[attacks] == k]) for k in names}

    mc = cb.copy()
    counts = {k: int(np.sum(types[mc] == k)) for k in names}
    missing = [k for k in names if counts[k] == 0]
    for k in missing:
        eligible = [i for i, row in enumerate(mc) if counts[types[row]] > 1]
        slot = int(rng.choice(eligible))
        counts[types[mc[slot]]] -= 1
        mc[slot] = pools[k][0]
        counts[k] += 1

    ordered = {}
    for k in names:
        kept = mc[types[mc] == k].tolist()
        keptset = set(kept)
        ordered[k] = kept + [int(v) for v in pools[k] if v not in keptset]
    quotas = dict.fromkeys(names, 0)
    remaining = n//2
    while remaining:
        active = [k for k in names if quotas[k] < len(pools[k])]
        share = max(1, remaining//len(active))
        for k in active:
            take = min(share, len(pools[k]) - quotas[k], remaining)
            quotas[k] += take
            remaining -= take
            if remaining == 0:
                break
    target = np.array([v for k in names for v in ordered[k][:quotas[k]]])
    targetset = set(target)
    tb = mc.copy()
    slots = [i for i, v in enumerate(tb) if v not in targetset]
    existing = set(tb)
    extra = [v for v in target if v not in existing]
    assert len(slots) == len(extra)
    tb[slots] = extra

    order = rng.permutation(n)
    out = {k: np.r_[benign, a][order] for k, a in zip(['balance', 'coverage', 'uniform'], [cb, mc, tb])}
    for idx in out.values():
        assert len(set(idx)) == n and np.sum(y[idx]) == n//2
        assert np.array_equal(idx[y[idx] == 0], out['balance'][y[out['balance']] == 0])
    assert np.sum(cb != mc) == len(missing)
    assert set(types[mc]) == set(names) and set(types[tb]) == set(names)
    return out


def stratified(y, cb, n, seed, attack_ratio):
    """ST: CB with k = |n/2 - round(n*ratio)| randomly chosen slots switched to records of the
    other class, drawn uniformly from records outside the subset, so that the subset keeps
    the training pool's attack ratio."""
    n_att = int(round(n*attack_ratio)); k = abs(n//2 - n_att)
    rng = np.random.default_rng(300000 + seed + 1000*n)
    st = cb.copy()
    if k:
        give = 1 if n_att < n//2 else 0          # class whose slots go to the other class
        slots = rng.choice(np.flatnonzero(y[cb] == give), k, replace=False)
        pool = np.setdiff1d(np.flatnonzero(y == 1 - give), cb)
        st[slots] = rng.choice(pool, k, replace=False)
    assert len(set(st)) == n and int(y[st].sum()) == n_att
    return st


def all_subsets(y, types, n, seed):
    """The four paired subsets of one dataset, n and seed."""
    out = paired_subsets(y, types, n, seed)
    out['stratified'] = stratified(y, out['balance'], n, seed, float(y.mean()))
    return out
