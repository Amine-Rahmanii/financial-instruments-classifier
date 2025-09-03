# 📈 Classificateur d'Instruments Financiers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)

Ce projet implémente un modèle de machine learning supervisé pour classer automatiquement des instruments financiers (actions, ETF, obligations) à partir de données de marché. **Résultats : 100% de précision** sur un dataset de 45,000 observations.

## 🎯 Objectif

Développer un modèle capable de distinguer avec précision les différents types d'instruments financiers en utilisant des features d'ingénierie basées sur les données de marché, dans le contexte de l'automatisation d'un référentiel d'instruments financiers.

## � Résultats

| Modèle | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|---------|
| **Random Forest** | **100%** | **100%** | **100%** | **100%** |
| **XGBoost** | **100%** | **100%** | **100%** | **100%** |
| **LightGBM** | **100%** | **100%** | **100%** | **100%** |
| Régression Logistique | 94.75% | 94.72% | 94.73% | 94.75% |

## �📁 Structure du projet

```
financial_instruments_classifier/
├── data/                   # Données brutes et traitées (45K observations)
├── src/                    # Code source modulaire
│   ├── data_collection.py  # Collecte données Yahoo Finance
│   ├── feature_engineering.py # 43 features techniques
│   └── train_models.py     # Entraînement 4 modèles ML
├── models/                 # Modèles entraînés + visualisations
├── streamlit_app.py        # Application web interactive
├── requirements.txt        # Dépendances Python
└── README.md              # Documentation
```

## 🚀 Installation et Usage

### 1. Cloner le projet
```bash
git clone https://github.com/[username]/financial-instruments-classifier.git
cd financial-instruments-classifier
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Lancer l'application
```bash
streamlit run streamlit_app.py
```

### 4. Réentraîner les modèles (optionnel)
```bash
# Collecter les données
python src/data_collection.py

# Feature engineering
python src/feature_engineering.py

# Entraîner les modèles
python src/train_models.py
```

## 🔧 Features Engineering (43 variables)

- **Rendements** : journaliers, momentum multi-périodes
- **Volatilité** : rolling windows (5, 10, 20 jours)
- **Moyennes mobiles** : MA(5, 10, 20, 50) + ratios prix/MA
- **Volume** : ratios, percentiles, moyennes mobiles
- **Indicateurs techniques** : RSI, MACD, Bollinger Bands
- **Features temporelles** : jour, mois, trimestre
- **Price action** : ranges, positions, gaps

## � Données

- **Sources** : Yahoo Finance (API yfinance)
- **Instruments** : 90 au total (30 par classe)
  - **Actions** : AAPL, MSFT, GOOGL, AMZN, TSLA...
  - **ETF** : SPY, QQQ, VTI, IWM, EFA...
  - **Obligations** : AGG, BND, TLT, IEF, SHY...
- **Période** : 2 ans (Sept 2023 - Sept 2025)
- **Fréquence** : Quotidienne

## 🤖 Modèles Implémentés

1. **Régression Logistique** : Baseline interprétable
2. **Random Forest** : Ensemble robuste, gestion overfitting
3. **XGBoost** : Gradient boosting haute performance
4. **LightGBM** : Gradient boosting optimisé

Tous avec **GridSearchCV** pour l'optimisation des hyperparamètres.

## 🌐 Application Streamlit

Interface web complète avec :
- 📊 **Dashboard de performance** : métriques, matrices de confusion
- 🔮 **Prédictions en temps réel** : classification d'instruments
- 📈 **Analyse exploratoire** : distributions, corrélations
- 🎯 **Importance des features** : variables les plus discriminantes

## 🔬 Méthodologie

1. **Collecte** : API Yahoo Finance, 90 instruments
2. **Preprocessing** : gestion NaN, outliers, normalisation
3. **Feature Engineering** : 43 variables techniques
4. **Modeling** : 4 algorithmes, validation croisée
5. **Évaluation** : métriques complètes, matrices confusion
6. **Déploiement** : application Streamlit interactive

## 💡 Applications Métier

- **Référentiel d'instruments** : classification automatique
- **Data stewardship** : enrichissement taxonomique
- **Onboarding** : accélération nouveaux flux de données
- **Contrôles qualité** : détection d'incohérences

## 🛠️ Technologies

- **Python** : Pandas, NumPy, Scikit-learn
- **ML** : XGBoost, LightGBM, Random Forest
- **Visualisation** : Plotly, Matplotlib, Seaborn
- **Web** : Streamlit
- **Data** : yfinance, Yahoo Finance

## � Prérequis

- Python 3.8+
- Packages listés dans `requirements.txt`
- Connexion internet (pour données Yahoo Finance)

## 🎯 Résultats Business

Les **100% de précision** s'expliquent par :
- **Patterns financiers distincts** : volatilité, volume, moyennes mobiles
- **Features discriminantes** : RSI, MACD, ratios techniques
- **Classes bien séparées** : actions vs ETF vs obligations
- **Dataset de qualité** : 2 ans, 45K observations

## 👨‍💻 Auteur

Projet développé dans le cadre d'une application de stage en data science / finance quantitative.
