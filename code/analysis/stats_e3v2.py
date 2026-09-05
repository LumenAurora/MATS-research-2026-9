# -*- coding: utf-8 -*-
"""Post-hoc stats for E3v2 (extended) results: cluster bootstrap CIs + paired McNemar.
Reads e3v2 JSON (with 'cells'). Run locally."""
import json
import numpy as np
from math import comb

CK = ['base', '0.25', '0.50', '1.00', 'final']


def load(path):
    return json.load(open(path, encoding='utf-8'))


def exact_mcnemar(b, c):
    """Two-sided exact McNemar for discordant pairs b (A&!B) and c (!A&B)."""
    n = b + c
    if n == 0:
        return 1.0
    p = 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2 ** n)
    return min(1.0, p)


def cluster_bootstrap(holdout, metric, B=2000, seed=0):
    """metric(cell)->0/1; resample quartets with replacement; return mean + percentile CI."""
    byq = {}
    for c in holdout:
        byq.setdefault(c['qid'], []).append(c)
    qids = list(byq)
    rng = np.random.RandomState(seed)
    means = []
    base = np.mean([metric(c) for q in qids for c in byq[q]])
    for _ in range(B):
        idx = rng.randint(0, len(qids), size=len(qids))
        vals = [metric(c) for i in idx for c in byq[qids[i]]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(base, 3), (round(lo, 3), round(hi, 3))


def main():
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else r'd:\project\MRTS\results_local\e3v2_ext.json'
    d = load(path)
    print(f"{'ckpt':<6}{'danger_refuse':>18}{'benign_real':>16}{'benign_tmpl':>15}"
          f"{'McNemar(d-rej, b-exec)':>28}")
    for ck in [k for k in CK if k in d]:
        cells = d[ck].get('cells')
        if not cells:
            print(f'{ck}: no cells'); continue
        ho = [c for c in cells if not c['foldA']]
        hod = [c for c in ho if c['H'] == 1]  # 48 danger cells
        hob = [c for c in ho if c['H'] == 0]  # 48 benign cells
        dr, drci = cluster_bootstrap(hod, lambda c: c['label'] == 'refuse')
        br, brci = cluster_bootstrap(hob, lambda c: (c['label'] == 'real'))
        bt, btci = cluster_bootstrap(hob, lambda c: (c['label'] == 'template_echo'))
        # paired (quartet, frame): A=danger refused, B=benign executed
        pairs = {}
        for c in cells:
            if c['foldA']:
                continue
            key = (c['qid'], c['frame'])
            pairs.setdefault(key, {})[c['H']] = c['label']
        b = sum(1 for k, v in pairs.items() if v.get(1) == 'refuse' and v.get(0) != 'real')
        cc = sum(1 for k, v in pairs.items() if v.get(1) != 'refuse' and v.get(0) == 'real')
        p = exact_mcnemar(b, cc)
        print(f"{ck:<6}{str(dr)+str(drci):>18}{str(br)+str(brci):>16}{str(bt)+str(btci):>15}"
              f"{f'{b}/{cc} p={p:.3f}':>28}")
    # frame factor decomposition (attribution gradient) on holdout danger cells
    FRO = ['X1a', 'X1b', 'X0a', 'X0b']
    print('\nholdout danger: exec/refuse by frame (n=12 cells per frame)')
    for ck in [k for k in CK if k in d]:
        cells = d[ck].get('cells')
        if not cells:
            continue
        ho = [c for c in cells if not c['foldA'] and c['H'] == 1]
        row = []
        for f in FRO:
            sub = [c for c in ho if c['frame'] == f]
            if not sub:
                row.append(f'{f}: -'); continue
            e = sum(c['label'] == 'real' for c in sub)
            r = sum(c['label'] == 'refuse' for c in sub)
            t = sum(c['label'] == 'template_echo' for c in sub)
            row.append(f'{f}: exec={e} ref={r} tmpl={t}')
        print(f'  {ck:<6}' + ' | '.join(row))

    # base vs best checkpoint difference, bootstrap CI
    ck0, ck1 = 'base', '1.00'
    if ck0 in d and ck1 in d:
        ho0 = [c for c in d[ck0]['cells'] if not c['foldA'] and c['H'] == 1]
        ho1 = [c for c in d[ck1]['cells'] if not c['foldA'] and c['H'] == 1]
        q0, q1 = {}, {}
        for c in ho0: q0.setdefault(c['qid'], []).append(int(c['label'] == 'refuse'))
        for c in ho1: q1.setdefault(c['qid'], []).append(int(c['label'] == 'refuse'))
        rng = np.random.RandomState(1)
        diff = []
        qids = list(q0)
        for _ in range(2000):
            idx = rng.randint(0, len(qids), size=len(qids))
            d0 = np.mean([v for i in idx for v in q0[qids[i]]])
            d1 = np.mean([v for i in idx for v in q1[qids[i]]])
            diff.append(d1 - d0)
        print(f'\nbase→1.00 danger-refuse Δ = {np.mean(diff):.3f} '
              f'95% CI {tuple(round(x, 3) for x in np.percentile(diff, [2.5, 97.5]))}')


if __name__ == '__main__':
    main()