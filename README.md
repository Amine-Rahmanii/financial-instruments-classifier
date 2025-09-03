# Classificateur d'Instruments Financiers

Ce projet implémente un modèle de machine learning supervisé pour classer automatiquement des instruments financiers (actions, ETF, obligations) à partir de données de marché.

## 🎯 Objectif

Développer un modèle capable de distinguer avec précision les différents types d'instruments financiers en utilisant des features d'ingénierie basées sur les données de marché.

## 📁 Structure du projet

```
financial_instruments_classifier/
├── data/                   # Données brutes et traitées
├── src/                    # Code source
├── models/                 # Modèles entraînés
├── notebooks/              # Notebooks Jupyter pour l'exploration
├── streamlit_app.py        # Application Streamlit
├── requirements.txt        # Dépendances
└── README.md              # Ce fichier
```

## 🚀 Installation

1. Cloner le projet
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📊 Utilisation

### 1. Collecte des données
```python
python src/data_collection.py
```

### 2. Entraînement des modèles
```python
python src/train_models.py
```

### 3. Lancement de l'application Streamlit
```bash
streamlit run streamlit_app.py
```

## 🔧 Features Engineering

Le projet génère automatiquement les features suivantes :
- Rendements journaliers
- Volatilité (rolling)
- Moyennes mobiles (20j, 50j)
- Ratio Volume/Moyenne
- Indicateurs techniques (RSI, MACD)

## 📈 Modèles implémentés

- Régression Logistique
- Random Forest
- XGBoost
- LightGBM

## 📊 Métriques d'évaluation

- Accuracy
- F1-Score
- Précision et Rappel
- Matrice de confusion
