# MidiBERT Judge

Affinage de MidiBERT (transformateur pré-entraîné) pour l'évaluation automatique de séquences musicales MIDI. Le modèle prédit un score entre 0 et 10 en fonction de la qualité musicale.

## Approche

Entraînement par corruption : les séquences MAESTRO sont corrompues de manière contrôlée (pitch, rythme, phrasing, expression), puis le modèle apprend à associer un score à chaque niveau de sévérité.

## Résultats de diagnostic

| Sévérité | Score attendu | Prédiction | Statut |
|----------|---------------|------------|--------|
| 0% | 10.00 | 10.07 | OK |
| 25% | 5.00 | 5.78 | OK |
| 50% | 2.93 | 4.17 | OK |
| 75% | 1.34 | 3.40 | ATTENTION |
| 100% | 0.00 | 2.85 | ATTENTION |

Le modèle excelle sur les corruptions légères et modérées, mais peine sur les extrêmes (zone d'effondrement aux niveaux 75% et 100%).

## Courbe d'entraînement

![Loss Curve](loss_curve.png)

La perte MSE diminue régulièrement sur 50 epochs, montrant que le modèle apprend. Il y a cependant une zone de plateau en fin d'entraînement, suggérant une saturation sur les cas difficiles.
