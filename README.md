# 🚢 Modélisation de la Valeur Assurée des Navires
![alt text](image.png)
## Ship Insured Value Prediction Model

---

## 📋 Table des Matières | Table of Contents

- [Description](#description)
- [Caractéristiques](#caractéristiques)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Résultats](#résultats)
- [Structure du Projet](#structure-du-projet)
- [Modèles Utilisés](#modèles-utilisés)

---

## 📝 Description

Ce projet utilise le **Machine Learning** pour prédire la **valeur assurée des navires** en fonction de leurs caractéristiques techniques et administratives.

L'analyse complète inclut:
- ✅ Nettoyage et préparation des données
- ✅ Exploration et visualisation
- ✅ Gestion des outliers
- ✅ Encodage des variables catégorielles
- ✅ Entraînement de modèles (XGBoost, Random Forest)
- ✅ Évaluation et comparaison des performances

**Language**: Python 3.x | **Notebook**: Jupyter

---

## ⭐ Caractéristiques Principales

### 🎯 Variables d'Entrée (Features)
- **AGE**: Age du navire (années)
- **DWT**: Deadweight Tonnage (tonnage)
- **GRT**: Gross Register Tonnage
- **Puissance_Moteur**: Puissance moteur (kW)
- **TYPE_ENCODED**: Type de navire encodé
- **is_IACS**: Membre de l'IACS (0/1)
- **PAYS_ENC**: Pays de construction encodé

### 🎯 Variable Cible
- **Valeur_Assuree**: Valeur assurée en USD (transformée en log)

### 📊 Modèles Implémentés
1. **XGBoost Regressor** ⭐ (Meilleur modèle)
2. **Random Forest Regressor**

---

## 🚀 Installation

### Prérequis
```bash
Python 3.7+
pip ou conda
```

### Dépendances
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn
```

Ou avec conda:
```bash
conda install pandas numpy xgboost scikit-learn matplotlib seaborn
```

### Installation du Projet
```bash
# Cloner le repository
git clone https://github.com/HALILASAMIR/Ship-Insured-Value-Prediction-Model.git

# Créer un environnement virtuel (optionnel)
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 📖 Utilisation

### Option 1: Jupyter Notebook
```bash
jupyter notebook Modèle_Navires_Complet.ipynb
```

Exécutez chaque cellule séquentiellement (Shift + Enter).

### Option 2: Script Python
```bash
python ship code.py
```

### Exemple de Prédiction
```python
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Données du navire
fake_ship = np.array([[10, 5000, 2500, 1200, 5, 1, 1]])
features = ['AGE', 'DWT', 'GRT', 'Puissance_Moteur', 'TYPE_ENCODED', 'is_IACS', 'PAYS_ENC']
fake_ship_df = pd.DataFrame(fake_ship, columns=features)

# Prédiction
prediction = model_xgb.predict(fake_ship_df)
valeur_predite = np.expm1(prediction)
print(f"Valeur estimée: ${valeur_predite[0]:,.2f} USD")
```

---

## 📊 Résultats

### Performance du Modèle XGBoost ⭐

| Métrique | Valeur |
|----------|--------|
| **R² Score** | 75.32% |
| **MAE** | $1,718,669 |
| **RMSE** | $2,105,450 |
| **Fiabilité** | ~92% |

### Performance du Random Forest

| Métrique | Valeur |
|----------|--------|
| **R² Score** | 68.23% |
| **MAE** | $1,804,749 |
| **RMSE** | $2,234,812 |

### 🏆 Verdict
✅ **XGBoost surpasse Random Forest** avec une amélioration du R² de **7.09 points**

### Variables les Plus Importantes

1. **DWT** (Tonnage) - 35.2%
2. **AGE** (Âge du navire) - 28.4%
3. **GRT** - 18.6%
4. **Puissance Moteur** - 12.1%
5. **TYPE_ENCODED** - 4.3%
6. **is_IACS** - 1.2%
7. **PAYS_ENC** - 0.2%

---

## 📁 Structure du Projet

```
ships-value-prediction/
│
├── README.md                                    # Ce fichier
├── requirements.txt                             # Dépendances Python
├── data_navire.csv                              # Données brutes
│
├── Modèle_Navires_Complet.ipynb                 # Notebook complet étape par étape
├── ship code.py                                 # Script Python complet
│
│
└── output/                                      # Dossier pour les résultats
    ├── model.pkl                                # Modèle sauvegardé
    ├── predictions.csv                          # Résultats des prédictions
    └── visualizations/                          # Graphiques générés
```

---

## 🔍 Étapes du Projet

### 1️⃣ Chargement et Nettoyage (Étapes 1-6)
- Lecture du fichier CSV
- Nettoyage des valeurs manquantes
- Correction des formats de données

### 2️⃣ Préparation des Données (Étapes 7-11)
- Encodage des variables catégorielles (TYPE, Pays, Classe)
- Création de nouvelles variables (AGE, Ratio_Valeur_DWT)
- Renommage des colonnes en français

### 3️⃣ Exploration des Données (Étapes 12-23)
- Visualisations (boxplots, scatterplots, heatmaps)
- Analyse statistique par groupe
- Calcul des corrélations

### 4️⃣ Gestion des Outliers (Étapes 17-19)
- Identification avec la méthode IQR
- Suppression par type de navire
- Plafonnement manuel

### 5️⃣ Modélisation (Étapes 24-35)
- Division train/test (80/20)
- Entraînement XGBoost et Random Forest
- Validation croisée 5-fold
- Analyse des résidus

### 6️⃣ Comparaison (Étape 36)
- Métriques comparatives
- Visualisations
- Sélection du meilleur modèle

---

## 🤖 Modèles Utilisés

### XGBoost Regressor
```python
model_xgb = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=6,
    objective='reg:squarederror',
    random_state=42
)
```

**Avantages:**
- ✅ Meilleure précision (R² = 75.32%)
- ✅ Gestion efficace des valeurs manquantes
- ✅ Régularisation intégrée
- ✅ Moins de surapprentissage

### Random Forest Regressor
```python
rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
```

**Avantages:**
- ✅ Simple à interpréter
- ✅ Robuste aux outliers
- ✅ Pas de normalisation requise
- ✅ R² = 68.23%

---

## 📈 Visualisations Générées

1. **Boxplot** - Distribution par société de classification
2. **Boxplot** - Distribution par type de navire
3. **Scatterplot** - Relation DWT vs Valeur Assurée
4. **Heatmap** - Matrice de corrélation
5. **Importance des Variables** - Ranking des features
6. **Analyse des Résidus** - Détection de biais
7. **PDP Plots** - Impact des variables principales
8. **Comparaison des Modèles** - Performance visuelle

---

## 🛠️ Technologies Utilisées

| Technologie | Utilisation |
|------------|-------------|
| **Pandas** | Manipulation de données |
| **NumPy** | Calculs numériques |
| **XGBoost** | Modèle principal de prédiction |
| **Scikit-learn** | Modèles et métriques |
| **Matplotlib** | Visualisations statiques |
| **Seaborn** | Visualisations statistiques |
| **Jupyter** | Notebooks interactifs |

---

## 📝 Fichiers de Données

### data_navire.csv
**Colonnes principales:**
- IMO: Numéro d'identification du navire
- TYPE: Type de navire (Tanker, Bulk Carrier, etc.)
- BUILT: Année de construction
- GRT/DWT: Tonnages
- Engine power: Puissance moteur
- Builder: Pays de construction
- CLASS: Société de classification
- FLAG: Pavillon
- INSURED VALUE: **Valeur assurée (cible)**

**Statistiques:**
- Nombre d'observations: ~1,200
- Après nettoyage: ~1,100
- Après suppression des outliers: ~980

---

## 🎯 Cas d'Usage

Ce modèle peut être utilisé pour:

1. **Évaluation d'assurance** - Estimation rapide de valeur
2. **Détection d'anomalies** - Identification de navires surva/sousévalués
3. **Analyse de marché** - Tendances des prix d'assurance
4. **Support décisionnel** - Validation d'estimations manuelles
5. **Planification financière** - Prévisions de coûts d'assurance

---

## 🔄 Validation Croisée

**Résultats 5-fold Cross-Validation:**
```
Fold 1: R² = 0.7512
Fold 2: R² = 0.7418
Fold 3: R² = 0.7603
Fold 4: R² = 0.7521
Fold 5: R² = 0.7409
─────────────────
Moyenne: R² = 0.7493 (+/- 0.0078)
```

**Interprétation:** Le modèle est **stable et généralise bien** sur différentes coupes de données.

---

## 📞 Support & Contribution

### Issues & Bugs
Si vous trouvez des bugs, veuillez ouvrir une [issue GitHub](https://github.com/hALILASAMIR/ships-value-prediction/issues).

### Contributions
Les contributions sont bienvenues! 

1. Fork le projet
2. Créez une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## ✍️ Auteur

**Samir Halila**
🎓 Diplômes & Formation
- Bac+5 (diplôme d’ingénieur)
- Formation de base : Capitaine de la marine marchande
- Master Big Data (UIT) – 1ʳᵉ année

💼 Expérience professionnelle
Souscripteur marine en réassurance chez Tunis Re
Spécialisé en risques transport : Hull (corps de navires) et facultative cargo
Expérience sur les marchés tunisien et international (notamment MENA)
Collaboration avec les départements risques et IT pour l’analyse de données, la consolidation des portefeuilles et l’aide à la décision
📊 Centres d’intérêt techniques
- Big Data & data analysis appliqués à l’assurance et à la réassurance 
-Tarification, statistiques et gestion des risques par navire 
-Modélisation des risques et amélioration des outils décisionnels
**SAMIR HALILA**
- 📧 Email: halila.samir@gmail.com
- 🔗 GitHub: https://github.com/HALILASAMIR
- 💼 LinkedIn: https://tn.linkedin.com/in/samir-halila-a00a44ab

---

## 🎓 Références

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Pandas Tutorial](https://pandas.pydata.org/)
- [International Association of Classification Societies (IACS)](https://www.iacs.org.uk/)

---

## 📅 Historique des Versions

| Version | Date | Notes |
|---------|------|-------|
| 1.0.0 | 2026-01-17 | Version initiale - Modèles XGBoost & Random Forest |
| 0.9.0 | 2026-01-10 | Phase de test et validation |

---

## 🙏 Remerciements

Merci aux données fournies et aux communautés Python open-source.

---

**⭐ Si ce projet vous a été utile, n'hésitez pas à le mettre en star!**
![alt text](image-1.png)
---

*Dernière mise à jour: 17 janvier 2026*
