# -*- coding: utf-8 -*-
"""E3 analysis: behavior + H/X probe dynamics across safety-SFT LoRA checkpoints.
Checkpoints: base(0-step from E1) -> ckpt_0.10 -> 0.25 -> 0.50 -> 1.00 -> final.
Question: does safety training delete the dangerous-content representation (H probe down),
or change the execution/ownership behavior (X-related behavior) while H stays decodable?
"""
import json, os, numpy as np, torch
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DIR = r'd:\project\MRTS\results_local'
BASE_JSON = os.path.join(DIR, 'e1_qwen35-4b-base.jsonl')
BASE_ACT = os.path.join(DIR, 'e1_qwen35-4b-base_acts.pt')
CKS = ['base', '0.10', '0.25', '0.50', '1.00', 'final']
frames = ['X1a', 'X1b', 'X0a', 'X0b']
HARM = ['operational', 'deceptive', 'malicious']

def auroc(y, s):
    if len(set(y)) < 2: return None
    return float(roc_auc_score(y, s))

def probe_auroc(X, y, seeds=(0, 1, 2, 3, 4)):
    vals = []
    for sd in seeds:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.4, stratify=y, random_state=sd)
        if len(set(y[tr])) < 2 or len(set(y[te])) < 2: continue
        clf = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], y[tr])
        v = auroc(y[te], clf.decision_function(X[te]))
        if v is not None: vals.append(v)
    return float(np.mean(vals)) if vals else None

def load(path):
    return [json.loads(l) for l in open(path, encoding='utf-8')]

def summarize(recs):
    nd = sum(1 for r in recs if r['H'] == 1)
    nb = sum(1 for r in recs if r['H'] == 0)
    def act(h):
        c = Counter(r['action_label'] for r in recs if r['H'] == h)
        real = c.get('real', 0) + c.get('trigger', 0)
        return real, c.get('refuse', 0), len([r for r in recs if r['H'] == h])
    dr, dref, dn = act(1)
    br, bref, bn = act(0)
    frame_refuse = {fr: sum(1 for r in recs if r['H'] == 1 and r['frame'] == fr
                            and r['action_label'] == 'refuse') for fr in frames}
    frame_real = {fr: sum(1 for r in recs if r['H'] == 1 and r['frame'] == fr
                          and r['action_label'] in ('real', 'trigger')) for fr in frames}
    cat_exec = {c: sum(1 for r in recs if r['H'] == 1 and r['cat'] == c
                       and r['action_label'] in ('real', 'trigger')) for c in HARM}
    return {'danger_real': dr, 'danger_refuse': dref, 'danger_n': dn,
            'benign_real': br, 'benign_refuse': bref, 'benign_n': bn,
            'frame_refuse': frame_refuse, 'frame_real': frame_real, 'cat_exec': cat_exec}

def main():
    rows = []
    for ck in CKS:
        if ck == 'base':
            recs = load(BASE_JSON); d = torch.load(BASE_ACT, map_location='cpu')
        else:
            fname = 'par_final' if ck == 'final' else ck
            recs = load(os.path.join(DIR, f'e3_{fname}.jsonl'))
            d = torch.load(os.path.join(DIR, f'e3_{fname}_acts.pt'), map_location='cpu')
        acts = d['acts'].numpy(); meta = d['meta']
        H = np.array([m['H'] for m in meta]); FR = np.array([m['frame'] for m in meta])
        s = summarize(recs)
        # probe at mid layers [0,8,16,24,32] capped
        L1 = acts.shape[1]
        h_aucs = []; x_aucs = []
        for li in [8, 16, 24]:
            if li >= L1: continue
            xx = acts[:, li]; xx = xx - xx.mean(0, keepdims=True)
            # H leave-one-frame-out
            lfo = []
            for fr in frames:
                te = FR == fr
                if len(set(H[~te])) < 2 or len(set(H[te])) < 2: continue
                clf = LogisticRegression(C=0.1, max_iter=2000).fit(xx[~te], H[~te])
                lfo.append(auroc(H[te], clf.decision_function(xx[te])))
            h_aucs.append(np.mean([v for v in lfo if v is not None]) if lfo else None)
            yx = (FR == 'X1a').astype(int)
            x_aucs.append(probe_auroc(xx, yx))
        s['H_probe_lfo'] = round(float(np.mean([v for v in h_aucs if v is not None])), 3)
        s['X_probe'] = round(float(np.mean([v for v in x_aucs if v is not None])), 3)
        s['checkpoint'] = ck
        rows.append(s)
        fam = '+'.join(f"{fr}:{s['frame_real'][fr]}" for fr in frames)
        cat = '+'.join(f"{c}:{s['cat_exec'][c]}" for c in HARM)
        print(f"[{ck}] danger_real={s['danger_real']}/{s['danger_n']} refuse={s['danger_refuse']} "
              f"| benign_real={s['benign_real']}/{s['benign_n']} refuse={s['benign_refuse']} "
              f"| H_probe_lfo={s['H_probe_lfo']} X_probe={s['X_probe']}\n"
              f"      frame_real(danger) {fam}\n      cat_exec(danger,{s['danger_n']}) {cat}")
    json.dump(rows, open(os.path.join(DIR, 'e3_dynamics.json'), 'w'), ensure_ascii=False, indent=1)
    print('saved -> e3_dynamics.json')

if __name__ == '__main__':
    main()