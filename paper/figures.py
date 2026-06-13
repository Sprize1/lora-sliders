"""Generate all figures for the paper."""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTPUT = "research/paper/figures"
os.makedirs(OUTPUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Color palette
C = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']


def fig1_cross_architecture():
    """Figure 1: Self-correction asymmetry across 5 models."""
    models = ['Qwen2.5\n1.5B', 'Qwen3\n1.7B', 'LFM2.5\n1.2B', 'Phi-4\nmini', 'SmolLM3\n3B']
    asymmetries = [-8, 0, 8, 25, 17]
    colors = [C[0]] * 5

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(models)), asymmetries, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models)
    ax.set_ylabel('Asymétrie de self-correction (pp)')
    ax.set_title('Figure 1: Asymétrie de self-correction selon le modèle')
    for bar, val in zip(bars, asymmetries):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5 if val >= 0 else val - 4,
                f'{val:+d}pp', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(-20, 35)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig1_cross_arch.png')
    plt.close()
    print("Fig1 done")


def fig2_interpolation_sweeps():
    """Figure 2: LoRA interpolation sweeps — 3 models."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # LFM2.5
    with open("research/experiment/results/lora_interp.json") as f:
        lfm = json.load(f)
    # Phi-4-mini
    with open("research/experiment/results/phi4_interp.json") as f:
        phi = json.load(f)
    # SmolLM3
    with open("research/experiment/results/smollm3_interp.json") as f:
        smol = json.load(f)

    for ax, data, name in zip(axes, [lfm, phi, smol], ['LFM2.5-1.2B (hybride)', 'Phi-4-mini (transformer)', 'SmolLM3-3B (transformer)']):
        alphas = [r['alpha'] for r in data if 'alpha' in r]
        asyms = [r['asymmetry_pp'] for r in data if 'asymmetry_pp' in r]
        ax.plot(alphas, asyms, 'o-', color=C[0], linewidth=2, markersize=8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=alphas[asyms.index(min(asyms, key=abs))] if any(abs(a) < 4 for a in asyms) else alphas[0],
                   color=C[1], linestyle=':', linewidth=0.8)
        ax.set_xlabel('α')
        ax.set_ylabel('Asymétrie (pp)')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Figure 2: Interpolation LoRA — 3 architectures', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig2_interpolation.png')
    plt.close()
    print("Fig2 done")


def fig3_bidirectional():
    """Figure 3: Bidirectional control — honesty, self-correction, refusal."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Honesty
    with open("research/experiment/results/negative_alpha.json") as f:
        neg = json.load(f)

    honesty_data = neg['honesty']
    sc_data = neg['self_correct']

    # Honesty plot
    alphas = [r['alpha'] for r in honesty_data]
    scores = [r['score'] * 100 for r in honesty_data]
    axes[0].plot(alphas, scores, 'o-', color=C[2], linewidth=2, markersize=8)
    axes[0].axhline(y=scores[alphas.index(0.0)], color='gray', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('α'); axes[0].set_ylabel('Honnêteté (%)')
    axes[0].set_title('Honnêteté (mensonge ↔ vérité)')
    axes[0].grid(True, alpha=0.3)

    # Self-correction plot
    alphas = [r['alpha'] for r in sc_data]
    scores = [r['score'] for r in sc_data]
    axes[1].plot(alphas, scores, 'o-', color=C[0], linewidth=2, markersize=8)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('α'); axes[1].set_ylabel('Asymétrie (pp)')
    axes[1].set_title('Self-correction (indulgent ↔ critique)')
    axes[1].grid(True, alpha=0.3)

    # Refusal (manual data from experiment)
    alpha_vals = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    refusal_rates = [0, 0, 25, 12, 88, 88, 100, 100]
    axes[2].plot(alpha_vals, refusal_rates, 'o-', color=C[3], linewidth=2, markersize=8)
    axes[2].axhline(y=12, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('α'); axes[2].set_ylabel('Taux de refus (%)')
    axes[2].set_title('Refus de tâche (compliant ↔ refus)')
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Figure 3: Contrôle bidirectionnel par α négatif/positif', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig3_bidirectional.png')
    plt.close()
    print("Fig3 done")


def fig4_decorrelation():
    """Figure 4: Decorrelation matrix between 6 adapters."""
    # Manual decorrelation data from experiments
    names = ['Self-corr', 'Honesty', 'Pirate', 'Syco', 'Confid.', 'Refusal']
    # Cos sim matrix (symmetric, diag≈1)
    matrix = np.array([
        [1.000, 0.000, -0.001, -0.000, -0.001, 0.000],
        [0.000, 1.000, 0.001, -0.001, 0.002, 0.000],
        [-0.001, 0.001, 1.000, 0.000, 0.001, 0.000],
        [-0.000, -0.001, 0.000, 1.000, 0.001, 0.000],
        [-0.001, 0.002, 0.001, 0.001, 1.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    ])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=9, fontweight='bold' if i == j else 'normal')

    plt.colorbar(im, ax=ax, label='Similarité cosinus')
    ax.set_title('Figure 4: Matrice de décorrélation — 6 adapteurs', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig4_decorrelation.png')
    plt.close()
    print("Fig4 done")


def fig5_radar():
    """Figure 5: Behavioral axes summary — control range per axis."""
    axes_names = ['Self-corr.', 'Honnêteté', 'Pirate', 'Sycophance', 'Verbosité', 'Refus']
    ranges = [108, 60, 90, 29, 78, 100]  # pp or % range

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(axes_names, ranges, color=[C[i % len(C)] for i in range(6)], edgecolor='white')
    ax.set_xlabel('Plage de contrôle (pp ou %)')
    ax.set_title('Figure 5: Amplitude de contrôle par axe comportemental', fontsize=13, fontweight='bold')
    for bar, val in zip(bars, ranges):
        ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, str(val), va='center', fontweight='bold')
    ax.set_xlim(0, 130)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig5_ranges.png')
    plt.close()
    print("Fig5 done")


def fig6_triangle():
    """Figure 6: Bar chart comparing methods on asymmetry, FP, and specificity."""
    methods = ['Base', 'SFT v1\n(ERROR only)', 'SFT v2\n(balanced)', 'DPO\n(contrastive)', 'LoRA\nα=0.25']
    asymmetry = [75, 0, 25, 0, 0]
    false_pos = [0, 100, 60, 0, 5]
    specificity = [75, 60, 25, 12, 42]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars1 = ax1.bar(x - width, asymmetry, width, label='Asymétrie (pp) — plus bas = mieux', color='#d62728', edgecolor='white')
    bars2 = ax1.bar(x, false_pos, width, label='Faux positifs (%) — plus bas = mieux', color='#ff7f0e', edgecolor='white')
    ax1.set_ylabel('Asymétrie / Faux positifs (plus bas = mieux)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    bars3 = ax2.bar(x + width, specificity, width, label='Spécificité (%) — plus haut = mieux', color='#1f77b4', edgecolor='white')
    ax2.set_ylabel('Spécificité (%)', color='#1f77b4')
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.9)

    # Highlight the winner
    ax1.axvspan(3.5, 4.5, alpha=0.08, color='green')
    ax1.text(4, 105, '★ Point\nPareto-optimal', ha='center', fontsize=9, fontweight='bold', color='green', va='top')

    ax1.set_title("Figure 6 : Comparaison des méthodes — triangle {asymétrie, FP, spécificité}", fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig6_triangle.png')
    plt.close()
    print("Fig6 done")


if __name__ == "__main__":
    fig1_cross_architecture()
    fig2_interpolation_sweeps()
    fig3_bidirectional()
    fig4_decorrelation()
    fig5_radar()
    fig6_triangle()
    print(f"\nAll figures saved to {OUTPUT}/")
