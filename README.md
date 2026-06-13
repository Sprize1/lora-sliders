# LoRA Behavioral Sliders

Quand un assistant LLM génère une erreur puis qu'on lui demande de la vérifier, il la corrige rarement. Présentez-lui la même erreur comme venant d'un utilisateur, et il la corrige presque toujours. Cette asymétrie, et cinq autres comportements de surface, sont pilotables via un mécanisme simple : l'interpolation linéaire d'un adapteur LoRA. Un seul scalaire α, variant de −1.5 à +1.5, déplace le comportement de façon continue et réversible.

## Résultat

Six comportements contrôlables indépendamment. Chaque adapteur est entraîné sur 80 à 600 exemples synthétiques. Tous les adapteurs sont décorrélés dans l'espace des poids (cos < 0.003).

| Axe | Plage | Bidirectionnel |
|---|---|---|
| Self-correction | −58pp ↔ +50pp | ✅ |
| Honnêteté | 30% ↔ 90% | ✅ |
| Refus de tâche | 0% ↔ 100% | ✅ |
| Pirate | normal ↔ Arrr! | ✅ |
| Sycophance | 14% → 43% | Partiel |
| Verbosité | 24 ↔ 102 mots | ✅ |

Le signe de α contrôle la direction : un adapteur entraîné sur des exemples de refus, à α = −1.0, rend le modèle totalement compliant. À α = +0.75, il refuse 100% des requêtes inappropriées. Un adapteur entraîné sur l'honnêteté, à α = −1.5, rend le modèle malhonnête — il nie savoir faire 2+2.

## Structure

```
├── paper/             # Papier (arXiv-ready LaTeX + PDF + figures)
├── experiment/        # Code et résultats
│   ├── data/          # Données d'entraînement (JSONL, ~15 fichiers)
│   ├── results/       # Résultats bruts (JSON, ~25 fichiers)
│   └── model_output_*/# Adapteurs LoRA (poids non versionnés)
├── .gitignore
└── README.md
```

## Reproduire

Matériel : GPU AMD avec ROCm (testé sur RX 7600 XT 16 Go) ou NVIDIA avec CUDA. Python 3.12.

```bash
uv venv --python 3.12 .venv312
source .venv312/Scripts/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1  # AMD
uv pip install torch torchvision  # NVIDIA
uv pip install transformers peft accelerate datasets trl safetensors matplotlib requests
```

Créer un `.env` à la racine avec une clé API DeepSeek (format Anthropic-compatible) :
```
ANTHROPIC_AUTH_TOKEN=sk-...
```

Puis lancer une expérience :
```bash
python experiment/personality_axes.py   # axes honnêteté + confiance
python experiment/refusal_axis.py       # axe refus de tâche
python experiment/final_validation.py   # validation N=50
```

Les adapteurs se génèrent en 2 à 30 minutes selon la taille du dataset.

## Modèles testés

LFM2.5-1.2B (hybride LIV-convolution), Phi-4-mini (transformer 3.8B), SmolLM3-3B (transformer), Qwen2.5-1.5B et Qwen3-1.7B (transformers).

## Citation

```bibtex
@misc{roy2026lora,
  title={Contrôle comportemental des LLMs par interpolation d'adapteurs LoRA},
  author={Oubayd Roy},
  year={2026}
}
```
