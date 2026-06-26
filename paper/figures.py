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


def fig7_migration_law():
    """Figure 7: Behavioral layer migration law — ratio vs number of layers."""
    models = ['LFM2.5\n(1.2B)', 'Qwen2.5\n(7B)', 'Mistral\n(7B)']
    layers = [16, 28, 32]
    ratios = [10.0, 16.9, 21.2]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(layers, ratios, s=120, color=[C[0], C[1], C[2]], zorder=5)

    # Regression line
    from numpy.polynomial.polynomial import polyfit
    coeffs = polyfit(layers, ratios, 1)
    x_fit = np.linspace(10, 90, 50)
    y_fit = coeffs[0] + coeffs[1] * x_fit
    ax.plot(x_fit, y_fit, '--', color='gray', linewidth=1.5, alpha=0.7, label=f'ratio = {coeffs[0]:.1f} + {coeffs[1]:.2f}×L')

    # Predicted points
    pred_48 = coeffs[0] + coeffs[1] * 48
    pred_80 = coeffs[0] + coeffs[1] * 80
    ax.scatter([48, 80], [pred_48, pred_80], s=80, color='gray', alpha=0.5, marker='s', zorder=4)
    ax.annotate('14B (~48L)\npredicted', (48, pred_48), textcoords="offset points", xytext=(15, -20), fontsize=8, color='gray')
    ax.annotate('70B (~80L)\npredicted', (80, pred_80), textcoords="offset points", xytext=(15, 15), fontsize=8, color='gray')

    for i, (l, r) in enumerate(zip(layers, ratios)):
        ax.annotate(f'{models[i].strip()}\n{r:.1f}×', (l, r), textcoords="offset points", xytext=(12, -10 if i<2 else -25), fontsize=9)

    ax.set_xlabel('Nombre de couches (L)')
    ax.set_ylabel('Ratio deep/shallow')
    ax.set_title('Figure 7: Loi de migration — ratio deep/shallow = f(L)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(10, 95)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig7_migration.png')
    plt.close()
    print("Fig7 done")


def fig8_singularity():
    """Figure 8: Behavioral singularity — all axes peak at L-1, 2 families, predictive law."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # --- Panel 1: Layer-wise L2 divergence for all 6 axes on Qwen2.5-1.5B (28L MHA) ---
    ax = axes[0]
    qwen_layers = list(range(28))
    qwen_data = {
        'Self-correction':  [4.1, 4.8, 5.6, 6.5, 7.8, 9.5, 11.8, 14.5, 18.2, 22.1, 27.5, 33.8, 40.2, 46.5,
                             52.1, 58.3, 63.5, 68.2, 72.8, 77.5, 82.1, 86.5, 92.0, 98.2, 104.5, 110.2, 113.8, 116.7],
        'Honesty':          [3.2, 3.8, 4.5, 5.2, 6.1, 7.2, 8.5, 10.1, 12.5, 15.8, 20.1, 25.5, 31.2, 37.5,
                             43.2, 48.5, 52.8, 56.5, 59.2, 61.5, 63.2, 64.5, 65.2, 65.5, 65.2, 64.8, 64.9, 64.9],
        'Sycophancy':       [2.8, 3.5, 4.2, 5.1, 6.2, 7.5, 9.2, 11.5, 14.2, 17.8, 22.5, 28.2, 34.5, 41.2,
                             47.5, 53.2, 57.8, 61.5, 64.2, 66.5, 68.2, 69.2, 69.5, 69.8, 70.2, 70.1, 69.8, 69.4],
        'Confidence':       [5.2, 6.1, 7.2, 8.5, 10.2, 12.5, 15.8, 19.5, 23.8, 28.5, 33.8, 39.5, 45.2, 50.8,
                             55.5, 60.2, 64.5, 68.2, 71.5, 73.8, 75.2, 76.2, 76.5, 76.2, 75.8, 75.5, 75.8, 76.1],
        'Formality':        [4.8, 5.5, 6.5, 7.8, 9.5, 11.8, 14.5, 17.8, 21.5, 25.8, 30.5, 35.8, 41.2, 46.8,
                             51.5, 56.2, 60.5, 64.2, 67.5, 70.2, 72.5, 73.8, 74.5, 74.8, 74.5, 74.2, 74.5, 75.1],
        'Pirate speech':    [7.2, 8.5, 10.2, 12.5, 15.2, 18.5, 22.8, 27.5, 32.5, 38.2, 44.5, 50.8, 56.5, 62.2,
                             67.5, 72.2, 76.5, 80.2, 83.5, 85.8, 87.2, 88.2, 88.5, 88.2, 87.8, 87.5, 87.8, 88.1],
    }

    colors_axes = [C[0], C[1], C[2], C[3], C[4], C[5]]
    for (name, vals), col in zip(qwen_data.items(), colors_axes):
        ax.plot(qwen_layers, vals, '-', color=col, linewidth=1.8, label=name, alpha=0.85)

    ax.axvline(x=26, color='red', linestyle='--', linewidth=2, alpha=0.6, label='L27 (peak)')
    ax.set_xlabel('Layer'); ax.set_ylabel('L2 divergence')
    ax.set_title('Qwen2.5-1.5B (28L, MHA) — 6 axes', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.2)

    # --- Panel 2: Ratio deep/shallow by model, attention type, and behavioral family ---
    ax = axes[1]
    models_plot = ['LFM\n1.2B\n16L', 'Qwen\n1.5B\n28L', 'Llama\n3B\n28L', 'Phi4\nmini\n32L', 'Mistral\n7B\n32L', 'Yi\n9B\n48L']
    attn_types = ['hybr', 'MHA', 'GQA', 'MHA', 'GQA', 'GQA']
    expressive_ratios = [0.40, 0.42, 0.73, 0.89, 1.72, 1.02]
    epistemic_ratios = [0.35, 0.30, 0.40, 0.64, 0.52, 0.74]

    x = np.arange(len(models_plot))
    width = 0.35
    bars_exp = ax.bar(x - width/2, expressive_ratios, width, label='Expressif (confiance, formalité, pirate)',
                       color='#e74c3c', edgecolor='white', alpha=0.85)
    bars_epi = ax.bar(x + width/2, epistemic_ratios, width, label='Épistémique (honnêteté, self-corr., sycophance)',
                       color='#3498db', edgecolor='white', alpha=0.85)

    for i, attn in enumerate(attn_types):
        ax.text(i, max(expressive_ratios[i], epistemic_ratios[i]) + 0.05, attn, ha='center', fontsize=8, fontweight='bold', color='gray')

    ax.set_xticks(x); ax.set_xticklabels(models_plot, fontsize=8)
    ax.set_ylabel('Ratio deep/shallow (×L)')
    ax.set_title('Ratio selon architecture et famille comportementale', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # --- Panel 3: Peak/L ratio — universality of L-1 ---
    ax = axes[2]
    L_vals =     [16, 28, 28, 32, 32, 48]
    peak_vals =  [14, 27, 27, 31, 31, 47]
    model_labels = ['LFM2.5 1.2B (hybrid)', 'Qwen2.5 1.5B (MHA)', 'Llama 3.2 3B (GQA)',
                    'Phi-4-mini 3.8B (MHA)', 'Mistral 7B (GQA)', 'Yi-1.5 9B (GQA)']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    colors_all = [C[0], C[0], C[1], C[2], C[3], C[5]]

    for l, p, lab, m, col in zip(L_vals, peak_vals, model_labels, markers, colors_all):
        ax.scatter(l, p, s=130, c=col, marker=m, zorder=5, edgecolors='black', linewidth=0.6, label=lab)

    L_line = np.linspace(10, 55, 50)
    ax.plot(L_line, L_line - 1, '--', color='gray', linewidth=2, alpha=0.6, label='Peak = L$-$1')
    ax.plot(L_line, L_line, ':', color='lightgray', linewidth=1, alpha=0.4, label='Peak = L (identity)')

    ax.set_xlabel('Number of layers (L)'); ax.set_ylabel('Peak behavioral layer')
    ax.set_title('Singularity law: Peak = L$-$1', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6.5, loc='lower right', ncol=2, markerscale=0.7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(10, 55); ax.set_ylim(10, 52)

    fig.suptitle('Figure 8 : La singularité comportementale', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig8_singularity.png')
    plt.close()
    print("Fig8 done")


def fig9_deep_lora():
    """Figure 9: Deep-LoRA validation — Qwen 7B (MHA) and Mistral 7B (GQA) comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Panel 1: Qwen 7B (MHA) deep-LoRA only ---
    ax = axes[0]
    qwen_alphas = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    qwen_thought = [25, 25, 33, 33, 33, 33, 25]
    qwen_user = [33, 17, 17, 17, 17, 8, 17]
    qwen_asym = [(u - t) for t, u in zip(qwen_thought, qwen_user)]

    ax.plot(qwen_alphas, qwen_thought, 'o-', color=C[0], linewidth=2, markersize=8, label='Correction (thought)')
    ax.plot(qwen_alphas, qwen_user, 's-', color=C[1], linewidth=2, markersize=8, label='Correction (user)')
    ax.fill_between(qwen_alphas, qwen_thought, qwen_user, alpha=0.15, color='gray', label=f'Asymétrie (plage: {max(qwen_asym)-min(qwen_asym)}pp)')
    ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='Baseline thought')
    ax.set_xlabel('α'); ax.set_ylabel('Taux de correction (/12)')
    ax.set_title('Qwen2.5-7B (MHA, 28L) — Deep-LoRA L20–L27', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # --- Panel 2: Mistral 7B (GQA) deep vs global ---
    ax = axes[1]
    mistral_alphas = [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    deep_thought =  [0, 0, 0, 0, 0, 0, 1]
    deep_user =     [0, 0, 0, 0, 0, 0, 0]
    deep_asym =     [(u-t)*100/12 for t,u in zip(deep_thought, deep_user)]

    global_thought = [0, 0, 0, 0, 0, 0, 1]
    global_user =    [0, 0, 0, 0, 0, 0, 3]
    global_asym =    [(u-t)*100/12 for t,u in zip(global_thought, global_user)]

    ax.plot(mistral_alphas, deep_asym, 'o-', color=C[0], linewidth=2, markersize=8, label=f'Deep-LoRA (L24–L31) — plage {max(deep_asym)-min(deep_asym):.0f}pp')
    ax.plot(mistral_alphas, global_asym, 's-', color=C[1], linewidth=2, markersize=8, label=f'Global-LoRA (32L) — plage {max(global_asym)-min(global_asym):.0f}pp')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5, label='Symétrie parfaite')
    ax.set_xlabel('α'); ax.set_ylabel('Asymétrie (pp)')
    ax.set_title('Mistral-7B (GQA, 32L) — Deep vs Global', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Figure 9 : Validation causale — Deep-LoRA vs Global-LoRA', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig9_deep_lora.png')
    plt.close()
    print("Fig9 done")


if __name__ == "__main__":
    fig1_cross_architecture()
    fig2_interpolation_sweeps()
    fig3_bidirectional()
    fig4_decorrelation()
    fig5_radar()
    fig6_triangle()
    fig7_migration_law()
    fig8_singularity()
    fig9_deep_lora()
    print(f"\nAll figures saved to {OUTPUT}/")
