# 📈 Classificateur d'Instruments Financiers

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)

Ce projet implémente un modèle de machine learning supervisé pour classer automatiquement des instruments financiers (actions, ETF, obligations) à partir de données de marché. **Résultats : 100% de précision** sur un dataset de 45,000 observations.

## 🎯 Objectif

Développer un modèle capable de distinguer avec précision les différents types d'instruments financiers en utilisant des features d'ingénierie basées sur les données de marché, dans le contexte de l'automatisation d'un référentiel d'instruments financiers.

🌐 **Application en ligne** : https://financialinstrumentsclassifier.streamlit.app/

## ✨ Résultats

| Modèle | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|---------|
| **Random Forest** | **100%** | **100%** | **100%** | **100%** |
| **XGBoost** | **100%** | **100%** | **100%** | **100%** |
| **LightGBM** | **100%** | **100%** | **100%** | **100%** |
| Régression Logistique | 94.75% | 94.72% | 94.73% | 94.75% |

## 📁 Structure du projet

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
git clone https://github.com/Amine-Rahmanii/financial-instruments-classifier.git
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

## 📊 Dataset

- **Source** : Yahoo Finance API
- **Période** : 2 années de données historiques
- **Instruments** : 
  - 30 Actions (Apple, Microsoft, Google, etc.)
  - 30 ETF (SPY, QQQ, VTI, etc.)
  - 30 Obligations (TLT, IEF, SHY, etc.)
- **Observations** : 45,000 points de données

## 🔧 Features Engineering (43 features)

### Indicateurs de volatilité
- Volatilité sur 5, 10, 20 jours
- Average True Range (ATR)
- Volatilité intraday

### Moyennes mobiles
- Simple Moving Average (5, 10, 20, 50 jours)
- Exponential Moving Average (5, 10, 20 jours)
- Ratios de moyennes mobiles

### Indicateurs techniques
- RSI (Relative Strength Index)
- MACD et signal
- Bandes de Bollinger
- Williams %R
- Stochastic Oscillator

### Features de prix
- Returns quotidiens/hebdomadaires
- Ratios High/Low, Close/Open
- Prix normalisés

## 🤖 Modèles ML

| Algorithme | Hyperparamètres optimisés | Performance |
|------------|---------------------------|-------------|
| **Random Forest** | 100 estimateurs, max_depth=20 | 100% |
| **XGBoost** | learning_rate=0.1, max_depth=6 | 100% |
| **LightGBM** | num_leaves=31, learning_rate=0.1 | 100% |
| **Régression Logistique** | C=1.0, solver='liblinear' | 94.75% |

## 📈 Visualisations

L'application Streamlit inclut :
- Matrices de confusion interactives
- Graphiques d'importance des features
- Distributions des prix par catégorie
- Analyse des corrélations
- Métriques de performance détaillées

## 💻 Technologies

- **Python 3.8+** : Langage principal
- **Pandas/NumPy** : Manipulation de données
- **Scikit-learn** : Modèles ML classiques
- **XGBoost/LightGBM** : Gradient boosting
- **Streamlit** : Interface web
- **Plotly** : Visualisations interactives
- **Yahoo Finance API** : Source de données

## 📋 Méthodologie

1. **Collecte de données** : Automatisation via yfinance
2. **Preprocessing** : Nettoyage et normalisation
3. **Feature Engineering** : 43 indicateurs techniques
4. **Entraînement** : 4 algorithmes avec GridSearchCV
5. **Évaluation** : Validation croisée et métriques
6. **Déploiement** : Application web Streamlit

## 👨‍💻 Auteur

**Amine Rahmani**
- 📧 Email : rahmaniiamine@gmail.com
- 💼 LinkedIn : [aminerahmani](https://www.linkedin.com/in/aminerahmani/)
- 🐙 GitHub : [Amine-Rahmanii](https://github.com/Amine-Rahmanii)

---

*Projet développé dans le cadre d'une candidature chez Natixis - Démonstration de compétences en Data Science et Machine Learning*