# -*- coding: utf-8 -*-
"""English-label figures for the ICLR paper."""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FIG = r'd:\project\MRTS\figures_en'
os.makedirs(FIG, exist_ok=True)
C1, C3, C4 = '#3498db', '#c0392b', '#27ae60'
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                     'axes.labelsize': 11, 'figure.dpi': 140})

# fig A: quartet design
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis('off')
ax.add_patch(plt.Rectangle((0.4, 3.4), 4.4, 2.5, fc='#fef9e7', ec='#b9770e', lw=1.5))
ax.add_patch(plt.Rectangle((5.2, 3.4), 4.4, 2.5, fc='#fdecea', ec='#c0392b', lw=1.5))
ax.add_patch(plt.Rectangle((0.4, 0.5), 4.4, 2.5, fc='#eaf2f8', ec='#2e86c1', lw=1.5))
ax.add_patch(plt.Rectangle((5.2, 0.5), 4.4, 2.5, fc='#fbe9e7', ec='#c0392b', lw=2.5))
ax.text(2.6, 6.2, 'Content H = benign', ha='center', fontsize=12, color='#148f77')
ax.text(7.4, 6.2, 'Content H = dangerous', ha='center', fontsize=12, color='#c0392b')
ax.text(0.5, 4.4, 'Plan quoted from others\n(dev2 / bot)', fontsize=10)
ax.text(5.3, 4.4, 'Others\' dangerous plan\n(e.g. dev2 suggests P)', fontsize=10, color='#a93226')
ax.text(0.5, 1.5, 'Own plan\n"my next step: P"', fontsize=10, color='#1b4f72')
ax.text(5.3, 1.6, 'Own dangerous plan\n(commit + confirm)', fontsize=10, color='#a93226', fontweight='bold')
ax.text(9.9, 0.7, 'X: ownership', ha='right', fontsize=10, color='#555')
ax.set_title('One experimental unit: identical token stream P, only content-risk and owner differ')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_design.png'), bbox_inches='tight'); plt.close()

# fig B: ownership gradient (dangerous cells)
frames = ['Own\n(committed)', 'User\n(discriminant)', 'dev2\nsuggests', 'dev2\ncron job']
execs, refuses = [21, 16, 13, 5], [1, 6, 9, 6]
x = np.arange(4); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.2))
b1 = ax.bar(x - w/2, execs, w, label='executed (of 24 dangerous)', color=C1)
b2 = ax.bar(x + w/2, refuses, w, label='refused (of 24 dangerous)', color=C3)
ax.set_xticks(x); ax.set_xticklabels(frames)
ax.set_ylabel('count / 24')
ax.set_title('Same dangerous plan, different owner: execution drops monotonically')
ax.legend()
for b in list(b1)+list(b2):
    ax.annotate(str(int(b.get_height())), (b.get_x()+b.get_width()/2, b.get_height()+0.3), ha='center', fontsize=9.5)
ax.annotate('structurally identical arm;\nonly the owner differs', xy=(1, 15.5), xytext=(1.0, 20.5),
            arrowprops=dict(arrowstyle='->', color='#8e44ad'), fontsize=9, color='#8e44ad')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_ownership_gradient.png'), bbox_inches='tight'); plt.close()

# fig C: E3v2-ext strict-eval dynamics (independent per-checkpoint load; content survives)
ck = ['base', '0.50', '1.00', 'final']
dang_ref = [0/48, 10/48, 36/48, 35/48]            # free-form refusal only
dang_noexec = [5/48, 47/48, 48/48, 48/48]          # refusal + template-refusal (no true execution)
benign_real = [40/48, 0/48, 0/48, 0/48]
probe_fixed = [0.888, 0.818, 0.839, 0.838]
fig, ax = plt.subplots(figsize=(7.6, 4.4))
ax.plot(ck, dang_noexec, '-o', color=C3, lw=2, label='dangerous not truly executed (refuse + template-refusal)')
ax.plot(ck, dang_ref, '-o', color='#e67e22', lw=1.6, ms=4, label='...of which free-form refusal')
ax.plot(ck, benign_real, '-s', color=C4, lw=2, label='benign truly executed (held-out quartets)')
ax.plot(ck, probe_fixed, '--', color='#3498db', lw=2.5, marker='^',
        label='fixed base content-probe AUROC (L24)')
ax.set_ylim(-0.03, 1.12); ax.set_ylabel('rate / AUROC')
ax.set_xlabel('safety-SFT progress →')
ax.set_title('Refusal generalizes to unseen plans, benign utility collapses,\nwhile dangerous content stays linearly readable (never deleted)')
ax.legend(loc='center right', fontsize=9)
ax.annotate('probe: 0.89 → 0.84 (no deletion evidence)', xy=(3, 0.84), xytext=(1.0, 0.60),
            arrowprops=dict(arrowstyle='->', color='#3498db'), fontsize=9.5, color='#1a5276')
plt.tight_layout(); plt.savefig(os.path.join(FIG, 'fig_h_survives.png'), bbox_inches='tight'); plt.close()

print('english figures ->', FIG, sorted(os.listdir(FIG)))