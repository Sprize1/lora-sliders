# Protocole expérimental : Asymétrie de self-correction

## Question de recherche
Peut-on fine-tuner un SLM pour qu'il traite SES propres erreurs avec la même
attention critique qu'il applique aux erreurs externes ?

## Hypothèse
Le fine-tuning sur des paires (erreur, correction) où la même erreur apparaît
dans les DEUX rôles (assistant/thought ET user) réduit l'asymétrie de correction.

## Design

### Pré-test (baseline)
1. Sélectionner N=100 questions réparties en 3 domaines : maths (40), logique (30), code (30)
2. Pour chaque question, générer une réponse erronée du modèle
3. Présenter l'erreur sous 2 conditions :
   - **Condition thought** : l'erreur apparaît comme une réponse précédente du modèle
     (rôle assistant dans le chat template)
   - **Condition user** : la même erreur apparaît comme venant d'un utilisateur
     (rôle user dans le chat template)
4. Demander au modèle d'identifier et corriger l'erreur
5. Mesurer :
   - Taux de correction condition thought : C_thought
   - Taux de correction condition user : C_user
   - Asymétrie = C_user - C_thought (attendu : 20-90pp positif)

### Training
1. Générer ~500 exemples d'entraînement via API DeepSeek
2. Chaque exemple : (question, raisonnement erroné, étapes de correction, réponse correcte)
3. Chaque exemple apparaît en 2 formats :
   - Format thought : erreur dans le rôle assistant, puis demande de correction
   - Format user : erreur dans le rôle user, puis demande de correction
4. Target identique : "L'erreur est... La correction est..."
5. Fine-tuning LoRA (r=16, alpha=32) sur Qwen2.5-1.5B-Instruct

### Post-test
1. Répéter le pré-test sur le modèle fine-tuné
2. Mesurer la nouvelle asymétrie
3. Test de généralisation : évaluer sur un 4ème domaine (raisonnement scientifique)
   ABSENT de l'entraînement

## Métriques
- **Métrique principale** : Δ asymétrie = asymétrie_pre - asymétrie_post
- **Métrique secondaire** : taux de correction global (moyenne thought + user)
- **Généralisation** : asymétrie sur le domaine hors-distribution

## Succès
- Réduction significative de l'asymétrie (>50% de réduction)
- Sans dégradation du taux de correction en condition user
- Transfert au domaine hors-distribution
