# Rapport d'Évaluation de la Prédiction du WER

Ce document synthétise les protocoles d'entraînement et d'évaluation menés sur les jeux de données publics (espagnol et multilingue), incluant le code exécuté, la qualité des corpus et l'ensemble des métriques de performance obtenues.

---

## 1. Entraînement et Évaluation sur Données Publiques

### 1.1 Train / Test sur les Données Publiques Espagnoles

#### Configuration et Code Associé

```python
import run_public as rp, json

# 1. Évaluation de la qualité des données (Block 2 - Espagnol)
pool, coverage = rp.load_pool(rp.PLAN_BLOCK2)
print(json.dumps(rp.wer_by_condition(pool), indent=1))
print(json.dumps(rp.proxy_blindness(pool), indent=1))
rp.save_report({
    "coverage": coverage,
    "wer": rp.wer_by_condition(pool),
    "blindness": rp.proxy_blindness(pool)
}, "block2_data")

# 2. Évaluation des métriques selon différents protocoles
summaries = rp.evaluate_pool(pool, protocols=("group", "lodo", "loco"))
rp.save_report(summaries, "block2_transferable")
```

#### Qualité des Données (Espagnol - Block 2)

| Corpus / Condition | $N$ | WER Moyen Pondéré | WER Moyen Segment | Écart-type ($\sigma$) | % WER = 0 | % pWER = 0 | % Aveugle (Blind) | Pearson ($r$) | Spearman ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fleurs / clean** | 200 | 0.0293 | 0.0295 | 0.0450 | 57.00% | 57.50% | 9.00% | 0.7348 | 0.7027 |
| **fleurs / noise_snr5** | 200 | 0.0638 | 0.0658 | 0.0751 | 34.50% | 26.50% | 4.50% | 0.8275 | 0.8080 |
| **mls / clean** | 200 | 0.0763 | 0.0838 | 0.1263 | 31.00% | 27.50% | 9.50% | 0.9571 | 0.7648 |
| **mls / noise_snr5** | 200 | 0.2292 | 0.2468 | 0.2435 | 5.50% | 3.00% | 0.50% | 0.2888 | 0.9250 |
| **voxpopuli / clean** | 200 | 0.0689 | 0.0676 | 0.1341 | 39.00% | 42.50% | 14.00% | 0.1852 | 0.4940 |
| **voxpopuli / noise_snr5** | 200 | 0.0914 | 0.0931 | 0.1551 | 29.00% | 40.50% | 17.00% | 0.3403 | 0.5471 |

#### Métriques d'Évaluation par Protocole (Group, LODO, LOCO)

*Ensemble de features : `transferable` ($22$ features) | Total segments : $1200$ | WER moyen : $0.0978 \pm 0.1602$ | WER corpus réel : $0.0958$*

| Protocole | Modèle | MAE | MAE (IC 95%) | RMSE | $R^2$ | Biais | Pearson ($r$) | Spearman ($\rho$) | WER Corpus Estimé | Gap Corpus | Gain vs pWER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group** | **mean** | 0.0994 | [0.0796, 0.1243] | 0.1623 | -0.0266 | 0.0000 | -0.2174 | -0.2061 | 0.0982 | 0.0024 | -0.964 |
| | **pwer** | 0.0506 | [0.0387, 0.0647] | 0.1976 | -0.5226 | 0.0078 | 0.6251 | 0.7483 | 0.0959 | 0.0001 | 0.000 |
| | **ridge** | 0.0376 | [0.0334, 0.0420] | 0.0602 | 0.8586 | 0.0013 | 0.9288 | 0.7285 | 0.0877 | 0.0082 | +0.257 |
| | **lasso** | 0.0368 | [0.0330, 0.0411] | 0.0528 | 0.8915 | 0.0000 | 0.9442 | 0.7136 | 0.0868 | 0.0090 | +0.273 |
| | **hgb** | 0.0338 | [0.0293, 0.0387] | 0.0521 | 0.8942 | 0.0019 | 0.9457 | 0.7884 | 0.0926 | 0.0033 | +0.332 |
| | **xgb** | 0.0360 | [0.0305, 0.0427] | 0.0594 | 0.8623 | -0.0091 | 0.9333 | 0.7788 | 0.0826 | 0.0132 | +0.289 |
| **LODO** | **mean** | 0.1055 | [0.0815, 0.1353] | 0.1695 | -0.1196 | -0.0000 | -0.3094 | -0.3157 | 0.0966 | 0.0008 | -1.085 |
| | **pwer** | 0.0506 | [0.0387, 0.0647] | 0.1976 | -0.5226 | 0.0078 | 0.6251 | 0.7483 | 0.0959 | 0.0001 | 0.000 |
| | **ridge** | 0.0418 | [0.0378, 0.0463] | 0.0629 | 0.8456 | -0.0069 | 0.9210 | 0.6308 | 0.0783 | 0.0175 | +0.174 |
| | **lasso** | 0.0395 | [0.0357, 0.0439] | 0.0569 | 0.8736 | -0.0029 | 0.9349 | 0.6765 | 0.0835 | 0.0123 | +0.219 |
| | **hgb** | 0.0507 | [0.0429, 0.0592] | 0.0807 | 0.7463 | 0.0058 | 0.8860 | 0.6218 | 0.0921 | 0.0037 | -0.002 |
| | **xgb** | 0.0478 | [0.0382, 0.0595] | 0.0842 | 0.7234 | -0.0159 | 0.8701 | 0.6677 | 0.0754 | 0.0204 | +0.055 |
| **LOCO** | **mean** | 0.1104 | [0.0877, 0.1357] | 0.1728 | -0.1642 | 0.0000 | -0.2339 | -0.2940 | 0.0981 | 0.0023 | -1.182 |
| | **pwer** | 0.0506 | [0.0387, 0.0647] | 0.1976 | -0.5226 | 0.0078 | 0.6251 | 0.7483 | 0.0959 | 0.0001 | 0.000 |
| | **ridge** | 0.0398 | [0.0355, 0.0445] | 0.0685 | 0.8173 | 0.0026 | 0.9051 | 0.7341 | 0.0907 | 0.0051 | +0.213 |
| | **lasso** | 0.0393 | [0.0355, 0.0434] | 0.0611 | 0.8544 | 0.0028 | 0.9245 | 0.7219 | 0.0902 | 0.0057 | +0.223 |
| | **hgb** | 0.0405 | [0.0358, 0.0452] | 0.0622 | 0.8492 | -0.0012 | 0.9252 | 0.7534 | 0.0891 | 0.0067 | +0.200 |
| | **xgb** | 0.0397 | [0.0320, 0.0490] | 0.0750 | 0.7807 | -0.0110 | 0.8935 | 0.7763 | 0.0818 | 0.0140 | +0.215 |

---

### 1.2 Train / Test sur les Données Publiques Multilangues

#### Configuration et Code Associé

```python
import run_public as rp, evaluate_public as ep, json

# 1. Évaluation de la qualité des données (Block 3 - Multilangue)
pool, coverage = rp.load_pool(rp.PLAN_BLOCK3)
print(json.dumps(rp.wer_by_condition(pool), indent=1))
print(json.dumps(rp.proxy_blindness(pool), indent=1))
rp.save_report({
    "coverage": coverage,
    "wer": rp.wer_by_condition(pool),
    "blindness": rp.proxy_blindness(pool)
}, "block3_data")

# 2. Évaluation des métriques sur protocoles group, lodo, loco, lolo
summaries = rp.evaluate_pool(pool, protocols=("group", "lodo", "loco", "lolo"))
rp.save_report(summaries, "block3_transferable")

# 3. Comparaison statistique des modèles en protocole LODO (bootstrap)
result = ep.run_protocol(pool, protocol="lodo")
comparison = {
    model: ep.compare_models(result, reference="pwer", challenger=model)
    for model in ("ridge", "lasso", "hgb", "xgb")
}
print(json.dumps(comparison, indent=1))

# 4. Décomposition de la variance (par condition et par corpus)
print(json.dumps(ep.metrics_by_stratum(result, key="condition", models=("pwer", "ridge", "xgb")), indent=1))
print(json.dumps(ep.metrics_by_stratum(result, key="corpus", models=("pwer", "ridge")), indent=1))
```

#### Qualité des Données (Multilangue - Block 3)

| Corpus / Condition | $N$ | WER Moyen Pondéré | WER Moyen Segment | Écart-type ($\sigma$) | % WER = 0 | % pWER = 0 | % Aveugle (Blind) | Pearson ($r$) | Spearman ($\rho$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fleurs / clean** | 1000 | 0.0517 | 0.0529 | 0.0708 | 43.90% | 41.90% | 9.20% | 0.6989 | 0.6424 |
| **fleurs / noise_snr5** | 1000 | 0.1453 | 0.1495 | 0.1546 | 18.90% | 13.60% | 2.50% | 0.7775 | 0.8575 |
| **mls / clean** | 800 | 0.0629 | 0.0666 | 0.0901 | 25.37% | 20.13% | 5.87% | 0.8021 | 0.7328 |
| **mls / noise_snr5** | 800 | 0.1770 | 0.1802 | 0.1813 | 7.62% | 5.12% | 1.50% | 0.2387 | 0.8890 |
| **voxpopuli / clean** | 800 | 0.0774 | 0.0764 | 0.1149 | 38.25% | 39.00% | 14.50% | 0.1113 | 0.4520 |
| **voxpopuli / noise_snr5** | 800 | 0.1136 | 0.1229 | 0.1648 | 25.87% | 33.00% | 13.50% | 0.4492 | 0.5980 |

#### Métriques Globales par Protocole (Group, LODO, LOCO, LOLO)

*Ensemble de features : `transferable` ($22$ features) | Total segments : $5200$ | WER moyen : $0.1075 \pm 0.1423$ | WER corpus réel : $0.1059$*

| Protocole | Modèle | MAE | MAE (IC 95%) | RMSE | $R^2$ | Biais | Pearson ($r$) | Spearman ($\rho$) | MAPE ($N=3795$) | Kendall ($\tau$) | C-index | Gap Corpus | Gain vs pWER |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Group** | **mean** | 0.0958 | [0.0882, 0.1040] | 0.1427 | -0.0061 | -0.0000 | -0.1039 | -0.1096 | 0.8518 | -0.0828 | 0.4611 | 0.0016 | -0.640 |
| | **pwer** | 0.0584 | [0.0507, 0.0671] | 0.1765 | -0.5387 | 0.0063 | 0.6043 | 0.7279 | 0.6269 | 0.5878 | 0.7956 | 0.0039 | 0.000 |
| | **ridge** | 0.0485 | [0.0451, 0.0521] | 0.0745 | 0.7258 | -0.0007 | 0.8520 | 0.7143 | 0.4633 | 0.5563 | 0.7893 | 0.0047 | +0.170 |
| | **lasso** | 0.0483 | [0.0452, 0.0516] | 0.0727 | 0.7389 | -0.0005 | 0.8596 | 0.7158 | 0.4632 | 0.5568 | 0.7898 | 0.0045 | +0.173 |
| | **hgb** | 0.0475 | [0.0444, 0.0506] | 0.0711 | 0.7500 | -0.0001 | 0.8660 | 0.7330 | 0.4510 | 0.5744 | 0.7991 | 0.0032 | +0.187 |
| | **xgb** | 0.0463 | [0.0427, 0.0500] | 0.0770 | 0.7074 | -0.0115 | 0.8451 | 0.7359 | 0.4479 | 0.5807 | 0.8024 | 0.0121 | +0.207 |
| **LODO** | **mean** | 0.0958 | [0.0880, 0.1043] | 0.1427 | -0.0064 | 0.0004 | -0.0738 | -0.0742 | 0.8429 | -0.0589 | 0.4748 | 0.0013 | -0.640 |
| | **pwer** | 0.0584 | [0.0507, 0.0671] | 0.1765 | -0.5387 | 0.0063 | 0.6043 | 0.7279 | 0.6269 | 0.5878 | 0.7956 | 0.0039 | 0.000 |
| | **ridge** | 0.0501 | [0.0468, 0.0535] | 0.0765 | 0.7110 | -0.0026 | 0.8437 | 0.6710 | 0.5042 | 0.5215 | 0.7717 | 0.0054 | +0.142 |
| | **lasso** | 0.0506 | [0.0473, 0.0539] | 0.0756 | 0.7173 | -0.0017 | 0.8470 | 0.6607 | 0.5108 | 0.5172 | 0.7670 | 0.0039 | +0.134 |
| | **hgb** | 0.0509 | [0.0477, 0.0543] | 0.0773 | 0.7046 | -0.0001 | 0.8406 | 0.6822 | 0.4942 | 0.5289 | 0.7755 | 0.0028 | +0.128 |
| | **xgb** | 0.0482 | [0.0445, 0.0520] | 0.0799 | 0.6847 | -0.0116 | 0.8316 | 0.7048 | 0.4729 | 0.5520 | 0.7873 | 0.0128 | +0.175 |
| **LOCO** | **mean** | 0.1134 | [0.1046, 0.1233] | 0.1608 | -0.2767 | -0.0000 | -0.3037 | -0.3414 | 1.0587 | -0.2873 | 0.3946 | 0.0018 | -0.942 |
| | **pwer** | 0.0584 | [0.0507, 0.0671] | 0.1765 | -0.5387 | 0.0063 | 0.6043 | 0.7279 | 0.6269 | 0.5878 | 0.7956 | 0.0039 | 0.000 |
| | **ridge** | 0.0500 | [0.0467, 0.0534] | 0.0770 | 0.7071 | 0.0005 | 0.8416 | 0.7096 | 0.4714 | 0.5512 | 0.7868 | 0.0031 | +0.144 |
| | **lasso** | 0.0504 | [0.0478, 0.0532] | 0.0725 | 0.7400 | 0.0021 | 0.8611 | 0.7070 | 0.4930 | 0.5600 | 0.7868 | 0.0004 | +0.137 |
| | **hgb** | 0.0516 | [0.0480, 0.0553] | 0.0799 | 0.6848 | 0.0025 | 0.8278 | 0.7220 | 0.4759 | 0.5633 | 0.7933 | 0.0006 | +0.116 |
| | **xgb** | 0.0513 | [0.0470, 0.0562] | 0.0897 | 0.6025 | -0.0098 | 0.7797 | 0.7279 | 0.4802 | 0.5710 | 0.7973 | 0.0094 | +0.122 |
| **LOLO** | **mean** | 0.0956 | [0.0882, 0.1038] | 0.1425 | -0.0032 | -0.0002 | -0.0743 | -0.0931 | 0.8484 | -0.0699 | 0.4677 | 0.0014 | -0.637 |
| | **pwer** | 0.0584 | [0.0507, 0.0671] | 0.1765 | -0.5387 | 0.0063 | 0.6043 | 0.7279 | 0.6269 | 0.5878 | 0.7956 | 0.0039 | 0.000 |
| | **ridge** | 0.0487 | [0.0454, 0.0520] | 0.0730 | 0.7364 | 0.0000 | 0.8582 | 0.7117 | 0.4663 | 0.5537 | 0.7882 | 0.0040 | +0.166 |
| | **lasso** | 0.0484 | [0.0453, 0.0516] | 0.0716 | 0.7467 | 0.0003 | 0.8642 | 0.7111 | 0.4657 | 0.5537 | 0.7882 | 0.0033 | +0.171 |
| | **hgb** | 0.0483 | [0.0452, 0.0515] | 0.0729 | 0.7378 | -0.0003 | 0.8590 | 0.7173 | 0.4561 | 0.5617 | 0.7925 | 0.0034 | +0.173 |
| | **xgb** | 0.0467 | [0.0430, 0.0506] | 0.0780 | 0.6996 | -0.0117 | 0.8406 | 0.7299 | 0.4515 | 0.5754 | 0.7997 | 0.0120 | +0.200 |

#### Comparaison des Modèles face au pWER (Bootstrap - Protocole LODO)

| Modèle Challenger (B) | Différence MAE ($\text{pWER} - \text{B}$) | IC Bas (95%) | IC Haut (95%) | En faveur du challenger ? (`favours_b`) |
| :--- | :---: | :---: | :---: | :---: |
| **ridge** | +0.0083 | 0.0026 | 0.0150 | **Oui** (Significatif) |
| **lasso** | +0.0078 | 0.0017 | 0.0151 | **Oui** (Significatif) |
| **hgb** | +0.0075 | 0.0016 | 0.0142 | **Oui** (Significatif) |
| **xgb** | +0.0102 | 0.0041 | 0.0175 | **Oui** (Significatif) |

#### Analyse de la Variance et Décomposition par Strate (LODO)

##### 1. Stratification par Condition Acoustique

| Strate (Condition) | $N$ | WER Moyen $\pm\ \sigma$ | Modèle | MAE | RMSE | $R^2$ | Biais | Pearson ($r$) | Spearman ($\rho$) | MAPE | Kendall ($\tau$) | C-index | Gap Corpus |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **clean** | 2600 | $0.0643 \pm 0.0926$ | **pwer** | 0.0540 | 0.1966 | -3.5056 | 0.0068 | 0.3401 | 0.5998 | 0.7867 | 0.4855 | 0.7450 | 0.0046 |
| | | *(WER réel: 0.0632)* | **ridge** | 0.0435 | 0.0662 | 0.4896 | 0.0025 | 0.7019 | 0.5160 | 0.5529 | 0.3943 | 0.7123 | 0.0013 |
| | | | **xgb** | 0.0393 | 0.0658 | 0.4958 | -0.0072 | 0.7085 | 0.5690 | 0.5088 | 0.4374 | 0.7350 | 0.0073 |
| **noise_snr5** | 2600 | $0.1508 \pm 0.1678$ | **pwer** | 0.0627 | 0.1537 | 0.1612 | 0.0058 | 0.7330 | 0.7828 | 0.5038 | 0.6257 | 0.8134 | 0.0034 |
| | | *(WER réel: 0.1485)* | **ridge** | 0.0566 | 0.0856 | 0.7401 | -0.0076 | 0.8621 | 0.7440 | 0.4666 | 0.5816 | 0.7957 | 0.0118 |
| | | | **xgb** | 0.0571 | 0.0919 | 0.7005 | -0.0161 | 0.8424 | 0.7640 | 0.4452 | 0.6016 | 0.8057 | 0.0179 |

##### 2. Stratification par Corpus

| Strate (Corpus) | $N$ | WER Moyen $\pm\ \sigma$ | Modèle | MAE | RMSE | $R^2$ | Biais | Pearson ($r$) | Spearman ($\rho$) | MAPE | Kendall ($\tau$) | C-index | Gap Corpus |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **fleurs** | 2000 | $0.1012 \pm 0.1296$ | **pwer** | 0.0416 | 0.0786 | 0.6323 | 0.0045 | 0.8445 | 0.8001 | 0.4825 | 0.6596 | 0.8343 | 0.0023 |
| | | *(WER réel: 0.0985)* | **ridge** | 0.0485 | 0.0649 | 0.7493 | 0.0184 | 0.8777 | 0.7959 | 0.5055 | 0.6356 | 0.8357 | 0.0158 |
| **mls** | 1600 | $0.1234 \pm 0.1540$ | **pwer** | 0.0470 | 0.1564 | -0.0308 | 0.0163 | 0.7409 | 0.8590 | 0.4637 | 0.7014 | 0.8532 | 0.0138 |
| | | *(WER réel: 0.1199)* | **ridge** | 0.0393 | 0.0580 | 0.8580 | 0.0055 | 0.9285 | 0.8392 | 0.4592 | 0.6755 | 0.8431 | 0.0025 |
| **voxpopuli** | 1600 | $0.0997 \pm 0.1439$ | **pwer** | 0.0907 | 0.2628 | -2.3337 | -0.0014 | 0.3710 | 0.5321 | 1.0099 | 0.4210 | 0.7076 | 0.0077 |
| | | *(WER réel: 0.0955)* | **ridge** | 0.0628 | 0.1019 | 0.4988 | -0.0368 | 0.7513 | 0.5291 | 0.5577 | 0.3908 | 0.7068 | 0.0413 |