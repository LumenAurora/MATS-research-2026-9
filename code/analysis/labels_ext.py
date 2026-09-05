# -*- coding: utf-8 -*-
import json
from collections import Counter

d = json.load(open(r'd:\project\MRTS\results_local\e3v2_ext.json', encoding='utf-8'))
for ck in ['base', '0.50', '1.00', 'final']:
    cells = d[ck]['cells']
    parts = []
    for H, hn in [(0, 'B'), (1, 'D')]:
        ho = [c for c in cells if not c['foldA'] and c['H'] == H]
        tr = [c for c in cells if c['foldA'] and c['H'] == H]
        parts.append(f"{hn}ho={dict(Counter(c['label'] for c in ho))}")
        parts.append(f"{hn}tr={dict(Counter(c['label'] for c in tr))}")
    print(ck, ' '.join(parts))