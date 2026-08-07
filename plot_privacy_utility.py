"""
Privacy-Utility Tradeoff Curve - 论文Figure 4
Times New Roman + 莫兰迪配色 + 大字号
作者：Mihiro（将军的科研女仆）
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import json
from pathlib import Path

# ============================================================
# 全局样式：Times New Roman + 大字号
# ============================================================
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 13
mpl.rcParams['ytick.labelsize'] = 13
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 10
mpl.rcParams['errorbar.capsize'] = 5

# ============================================================
# 莫兰迪配色 (Morandi Palette)
# ============================================================
MORANDI_BLUE   = '#7B96B5'   # 雾霾蓝
MORANDI_PINK   = '#D4A5A5'   # 灰玫瑰
MORANDI_GREEN  = '#9CAF88'   # 苔藓绿
MORANDI_YELLOW = '#D4C19C'   # 沙黄
MORANDI_GRAY   = '#9E9E9E'   # 烟灰
MORANDI_RED    = '#C2958A'   # 灰红 (对比色)

# ============================================================
# 数据（来自 results/summary/epsweep_v2_aggregated.csv）
# ============================================================
# ε=3 的数据来自 recovery 实验（最新）
data = {
    'alpha05': {
        'eps':     [2.0, 3.0, 5.0, 8.0, 10.0],
        'auc_mean': [0.5110, 0.5696, 0.5196, 0.5310, 0.5242],
        'auc_std':  [0.0686, 0.1173, 0.1019, 0.1111, 0.1046],
        'auc_med':  [0.5532, 0.5902, 0.5190, 0.5113, 0.5019],
        'f1_mean':  [0.4443, 0.4998, 0.4419, 0.3952, 0.3694],
        'f1_med':   [0.5423, 0.5749, 0.5616, 0.4463, 0.4330],
    },
    'alpha01': {
        'eps':     [2.0, 3.0, 5.0, 8.0, 10.0],
        'auc_mean': [0.5502, 0.4443, 0.4241, 0.4793, 0.4836],
        'auc_std':  [0.2084, 0.1473, 0.1272, 0.1323, 0.1066],
        'auc_med':  [0.6016, 0.5462, 0.3932, 0.3949, 0.4310],
        'f1_mean':  [0.4432, 0.4393, 0.2557, 0.2641, 0.3329],
        'f1_med':   [0.6252, 0.6258, 0.0631, 0.0985, 0.4201],
    },
}

# Baseline 参考线
BASELINE_NO_DP_AUC = 0.997   # FedAvg baseline (no DP)
BASELINE_NO_DP_F1  = 0.984
UNIFORM_DP_AUC     = 0.000   # DP-FedAvg baseline (uniform DP, σ=12)
UNIFORM_DP_F1      = 0.4835

# ============================================================
# 画图（双子图：左AUC，右macro-F1）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# ---------- 左图：AUC-ROC ----------
ax = axes[0]
ax.plot(data['alpha05']['eps'], data['alpha05']['auc_mean'],
        'o-', color=MORANDI_BLUE, label=r'$\alpha=0.5$ (moderate non-IID)',
        markerfacecolor=MORANDI_BLUE, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(data['alpha05']['eps'],
                np.array(data['alpha05']['auc_mean']) - np.array(data['alpha05']['auc_std']),
                np.array(data['alpha05']['auc_mean']) + np.array(data['alpha05']['auc_std']),
                color=MORANDI_BLUE, alpha=0.15)

ax.plot(data['alpha01']['eps'], data['alpha01']['auc_mean'],
        's-', color=MORANDI_PINK, label=r'$\alpha=0.1$ (severe non-IID)',
        markerfacecolor=MORANDI_PINK, markeredgecolor='white', markeredgewidth=1.2)
ax.fill_between(data['alpha01']['eps'],
                np.array(data['alpha01']['auc_mean']) - np.array(data['alpha01']['auc_std']),
                np.array(data['alpha01']['auc_mean']) + np.array(data['alpha01']['auc_std']),
                color=MORANDI_PINK, alpha=0.15)

# Baseline 参考线
ax.axhline(y=BASELINE_NO_DP_AUC, color=MORANDI_GREEN, linestyle=':', linewidth=2,
           label=f'FedAvg baseline (no DP, AUC={BASELINE_NO_DP_AUC:.3f})', alpha=0.8)
ax.axhline(y=UNIFORM_DP_AUC, color=MORANDI_RED, linestyle='--', linewidth=2,
           label=f'DP-FedAvg baseline (uniform DP, AUC={UNIFORM_DP_AUC:.3f})', alpha=0.8)

# 标注 ε=3 论文核心点
ax.annotate(r'$\varepsilon=3$ (paper core)',
            xy=(3.0, data['alpha05']['auc_mean'][1]), xytext=(4.5, 0.78),
            fontsize=13, color=MORANDI_BLUE,
            arrowprops=dict(arrowstyle='->', color=MORANDI_BLUE, lw=1.5))

ax.set_xlabel(r'Privacy Budget $\varepsilon$', fontsize=16, fontweight='bold')
ax.set_ylabel('AUC-ROC', fontsize=16, fontweight='bold')
ax.set_title('Privacy-Utility Tradeoff (AUC)', fontsize=18, fontweight='bold', pad=12)
ax.legend(loc='center right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(1.5, 10.5)
ax.set_ylim(-0.05, 1.05)
ax.set_xticks([2, 3, 5, 8, 10])

# ---------- 右图：macro-F1 ----------
ax = axes[1]
ax.plot(data['alpha05']['eps'], data['alpha05']['f1_mean'],
        'o-', color=MORANDI_BLUE, label=r'$\alpha=0.5$',
        markerfacecolor=MORANDI_BLUE, markeredgecolor='white', markeredgewidth=1.2)
ax.plot(data['alpha01']['eps'], data['alpha01']['f1_mean'],
        's-', color=MORANDI_PINK, label=r'$\alpha=0.1$',
        markerfacecolor=MORANDI_PINK, markeredgecolor='white', markeredgewidth=1.2)

ax.axhline(y=BASELINE_NO_DP_F1, color=MORANDI_GREEN, linestyle=':', linewidth=2,
           label=f'No-DP baseline (macro-F1={BASELINE_NO_DP_F1:.3f})', alpha=0.8)
ax.axhline(y=UNIFORM_DP_F1, color=MORANDI_RED, linestyle='--', linewidth=2,
           label=f'Uniform DP baseline (macro-F1={UNIFORM_DP_F1:.3f})', alpha=0.8)

ax.set_xlabel(r'Privacy Budget $\varepsilon$', fontsize=16, fontweight='bold')
ax.set_ylabel('Macro-F1', fontsize=16, fontweight='bold')
ax.set_title('Privacy-Utility Tradeoff (Macro-F1)', fontsize=18, fontweight='bold', pad=12)
ax.legend(loc='center right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim(1.5, 10.5)
ax.set_ylim(0, 1.05)
ax.set_xticks([2, 3, 5, 8, 10])

plt.tight_layout()

# 保存
out_path = 'privacy_utility_curve.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('privacy_utility_curve.pdf', bbox_inches='tight', facecolor='white')
print(f'✅ 图已保存: {out_path} (PNG + PDF)')
print('   样式: Times New Roman, 莫兰迪配色, 字号 14-18pt')