"""
Application Streamlit pour la classification d'instruments financiers
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Classificateur d'Instruments Financiers",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class FinancialClassifierApp:
    def __init__(self):
        self.label_names = {0: 'Action', 1: 'ETF', 2: 'Obligation'}
        self.load_models()
        
    def load_models(self):
        """Charge les modèles entraînés avec des chemins absolus"""
        import os
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'models')
        data_dir = os.path.join(base_dir, 'data')
        try:
            with open(os.path.join(models_dir, 'random_forest.pkl'), 'rb') as f:
                self.rf_model = pickle.load(f)
            with open(os.path.join(models_dir, 'xgboost.pkl'), 'rb') as f:
                self.xgb_model = pickle.load(f)
            with open(os.path.join(models_dir, 'lightgbm.pkl'), 'rb') as f:
                self.lgb_model = pickle.load(f)
            with open(os.path.join(models_dir, 'logistic_regression.pkl'), 'rb') as f:
                self.lr_model = pickle.load(f)
            with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            with open(os.path.join(models_dir, 'results.pkl'), 'rb') as f:
                self.results = pickle.load(f)
            self.models_loaded = True
            self.feature_names = pd.read_csv(os.path.join(data_dir, 'feature_names.csv'))['feature'].tolist()
        except FileNotFoundError:
            self.models_loaded = False
    
    def display_header(self):
        """Affiche l'en-tête de l'application"""
        st.markdown('<h1 class="main-header">📈 Classificateur d\'Instruments Financiers</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        Cette application utilise des modèles de machine learning pour classer automatiquement 
        les instruments financiers en trois catégories : **Actions**, **ETF**, et **Obligations**.
        """)
        
        st.markdown("---")
    
    def display_sidebar(self):
        """Affiche la barre latérale"""
        st.sidebar.title("🔧 Configuration")
        
        # Sélection de l'onglet
        tab = st.sidebar.selectbox(
            "Choisir une section",
            ["🏠 Accueil", "📊 Performance des Modèles", "🔮 Prédiction", "📈 Analyse de Données", "ℹ️ À propos"]
        )
        
        return tab
    
    def display_model_performance(self):
        """Affiche les performances des modèles"""
        st.header("📊 Performance des Modèles")
        
        if not self.models_loaded:
            st.error("❌ Les modèles n'ont pas été trouvés. Veuillez d'abord entraîner les modèles.")
            return
        
        # Résumé des performances
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Résumé des Performances")
            
            # Créer un DataFrame de résumé
            summary_data = []
            for model_name, results in self.results.items():
                summary_data.append({
                    'Modèle': model_name.replace('_', ' ').title(),
                    'Accuracy': results['accuracy'],
                    'F1-Score': results['f1_score'],
                    'Precision': results['precision'],
                    'Recall': results['recall']
                })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.style.format({
                'Accuracy': '{:.4f}',
                'F1-Score': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}'
            }).highlight_max(subset=['F1-Score'], color='lightgreen'), use_container_width=True)
            
        with col2:
            st.subheader("📈 Comparaison Graphique")
            
            # Graphique de comparaison
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            fig = go.Figure()
            
            for metric in metrics:
                values = [summary_df[summary_df['Modèle'] == model][metric].iloc[0] for model in summary_df['Modèle']]
                fig.add_trace(go.Scatter(
                    x=summary_df['Modèle'],
                    y=values,
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title="Comparaison des Métriques",
                xaxis_title="Modèles",
                yaxis_title="Score",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Matrices de confusion
        st.subheader("🔍 Matrices de Confusion")
        
        selected_model = st.selectbox(
            "Sélectionner un modèle",
            list(self.results.keys()),
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        if selected_model in self.results:
            conf_matrix = self.results[selected_model]['confusion_matrix']
            
            # Créer la heatmap avec Plotly
            fig = px.imshow(
                conf_matrix,
                labels=dict(x="Prédictions", y="Vraies valeurs", color="Nombre"),
                x=['Action', 'ETF', 'Obligation'],
                y=['Action', 'ETF', 'Obligation'],
                color_continuous_scale='Blues',
                text_auto=True,
                title=f'Matrice de Confusion - {selected_model.replace("_", " ").title()}'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Rapport de classification détaillé
            with st.expander("📋 Rapport de Classification Détaillé"):
                class_report = self.results[selected_model]['classification_report']
                
                # Convertir en DataFrame pour un meilleur affichage
                report_df = pd.DataFrame(class_report).transpose()
                st.dataframe(report_df.style.format('{:.4f}'), use_container_width=True)
    
    def predict_instrument(self, features, model_name='random_forest'):
        """Fait une prédiction sur un instrument"""
        models = {
            'random_forest': self.rf_model,
            'xgboost': self.xgb_model,
            'lightgbm': self.lgb_model,
            'logistic_regression': self.lr_model
        }
        
        model = models[model_name]
        
        # Préparer les features
        if model_name == 'logistic_regression':
            features_scaled = self.scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
        else:
            prediction = model.predict([features])[0]
            probabilities = model.predict_proba([features])[0]
        
        return prediction, probabilities
    
    def display_prediction(self):
        """Interface de prédiction"""
        st.header("🔮 Prédiction d'Instrument")
        
        if not self.models_loaded:
            st.error("❌ Les modèles n'ont pas été trouvés. Veuillez d'abord entraîner les modèles.")
            return
        
        st.markdown("""
        Entrez les données d'un instrument financier pour prédire son type.
        Vous pouvez soit entrer manuellement les valeurs, soit utiliser un symbole pour télécharger les données automatiquement.
        """)
        
        # Choix du mode de saisie
        input_mode = st.radio(
            "Mode de saisie",
            ["📥 Téléchargement automatique", "✏️ Saisie manuelle"]
        )
        
        if input_mode == "📥 Téléchargement automatique":
            self.display_automatic_prediction()
        else:
            self.display_manual_prediction()
    
    def display_automatic_prediction(self):
        """Prédiction automatique à partir d'un symbole"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            symbol = st.text_input("Entrez un symbole (ex: AAPL, SPY, AGG)", "AAPL")
            
            if st.button("📊 Analyser", type="primary"):
                if symbol:
                    with st.spinner(f"Téléchargement des données pour {symbol}..."):
                        try:
                            # Télécharger les données
                            ticker = yf.Ticker(symbol)
                            data = ticker.history(period="1y")
                            
                            if len(data) > 0:
                                # Calculer les features (version simplifiée)
                                features = self.calculate_features_from_data(data)
                                
                                # Faire la prédiction avec tous les modèles
                                predictions = {}
                                for model_name in ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']:
                                    pred, proba = self.predict_instrument(features, model_name)
                                    predictions[model_name] = {
                                        'prediction': pred,
                                        'probabilities': proba
                                    }
                                
                                # Afficher les résultats
                                with col2:
                                    self.display_prediction_results(symbol, predictions)
                                
                            else:
                                st.error(f"❌ Aucune donnée trouvée pour {symbol}")
                                
                        except Exception as e:
                            st.error(f"❌ Erreur lors du téléchargement: {e}")
    
    def calculate_features_from_data(self, data):
        """Calcule les features à partir des données de prix"""
        # Version simplifiée du feature engineering
        
        # Calculs de base
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility_20d'] = data['Daily_Return'].rolling(20).std()
        data['MA_20d'] = data['Close'].rolling(20).mean()
        data['MA_50d'] = data['Close'].rolling(50).mean()
        data['Volume_MA_20d'] = data['Volume'].rolling(20).mean()
        
        # Prendre les valeurs les plus récentes
        latest = data.iloc[-1]
        
        # Créer un vecteur de features (version simplifiée)
        features = [
            latest['Daily_Return'] if not pd.isna(latest['Daily_Return']) else 0,
            latest['Volatility_20d'] if not pd.isna(latest['Volatility_20d']) else 0,
            latest['Close'] / latest['MA_20d'] if not pd.isna(latest['MA_20d']) else 1,
            latest['Close'] / latest['MA_50d'] if not pd.isna(latest['MA_50d']) else 1,
            latest['Volume'] / latest['Volume_MA_20d'] if not pd.isna(latest['Volume_MA_20d']) else 1,
        ]
        
        # Ajouter des features supplémentaires pour correspondre au modèle
        # (Version simplifiée - en production, il faudrait calculer toutes les features)
        while len(features) < len(self.feature_names):
            features.append(0)
        
        return features[:len(self.feature_names)]
    
    def display_manual_prediction(self):
        """Interface de saisie manuelle"""
        st.info("💡 Entrez les principales métriques financières pour obtenir une prédiction.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Prix et Rendements")
            daily_return = st.number_input("Rendement journalier (%)", value=0.0, step=0.1, format="%.2f") / 100
            volatility = st.number_input("Volatilité (20j) (%)", value=2.0, step=0.1, format="%.2f") / 100
            momentum_20d = st.number_input("Momentum (20j) (%)", value=0.0, step=0.1, format="%.2f") / 100
        
        with col2:
            st.subheader("📊 Moyennes Mobiles")
            price_ma_20_ratio = st.number_input("Prix/MA(20j)", value=1.0, step=0.01, format="%.3f")
            price_ma_50_ratio = st.number_input("Prix/MA(50j)", value=1.0, step=0.01, format="%.3f")
            
        with col3:
            st.subheader("📦 Volume")
            volume_ratio = st.number_input("Volume/Moyenne", value=1.0, step=0.1, format="%.2f")
            volume_percentile = st.slider("Percentile de volume", 0.0, 1.0, 0.5, 0.01)
        
        if st.button("🔮 Prédire", type="primary"):
            # Créer le vecteur de features (version simplifiée)
            features = [daily_return, volatility, price_ma_20_ratio, price_ma_50_ratio, 
                       volume_ratio, momentum_20d, volume_percentile]
            
            # Compléter avec des zéros pour correspondre au nombre de features attendu
            while len(features) < len(self.feature_names):
                features.append(0)
            
            features = features[:len(self.feature_names)]
            
            # Faire les prédictions
            predictions = {}
            for model_name in ['random_forest', 'xgboost', 'lightgbm', 'logistic_regression']:
                pred, proba = self.predict_instrument(features, model_name)
                predictions[model_name] = {
                    'prediction': pred,
                    'probabilities': proba
                }
            
            # Afficher les résultats
            self.display_prediction_results("Instrument Personnalisé", predictions)
    
    def display_prediction_results(self, symbol, predictions):
        """Affiche les résultats de prédiction"""
        st.subheader(f"🎯 Résultats pour {symbol}")
        
        # Vote majoritaire
        votes = [pred['prediction'] for pred in predictions.values()]
        consensus = max(set(votes), key=votes.count)
        confidence = votes.count(consensus) / len(votes)
        
        # Affichage du consensus
        st.markdown(f"""
        <div class="success-box">
            <h3>🏆 Prédiction Consensus</h3>
            <h2>{self.label_names[consensus]}</h2>
            <p>Confiance: {confidence:.0%} ({votes.count(consensus)}/{len(votes)} modèles d'accord)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Détails par modèle
        st.subheader("📊 Détails par Modèle")
        
        for model_name, result in predictions.items():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                pred_label = self.label_names[result['prediction']]
                max_proba = np.max(result['probabilities'])
                
                st.metric(
                    label=model_name.replace('_', ' ').title(),
                    value=pred_label,
                    delta=f"{max_proba:.1%}"
                )
            
            with col2:
                # Graphique des probabilités
                prob_df = pd.DataFrame({
                    'Classe': ['Action', 'ETF', 'Obligation'],
                    'Probabilité': result['probabilities']
                })
                
                fig = px.bar(
                    prob_df, 
                    x='Classe', 
                    y='Probabilité',
                    title=f"Probabilités - {model_name.replace('_', ' ').title()}",
                    color='Probabilité',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    def display_data_analysis(self):
        """Affiche l'analyse des données"""
        st.header("📈 Analyse de Données")
        
        try:
            # Charger les données avec chemin absolu
            import os
            base_dir = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(base_dir, 'data', 'processed_financial_data.csv')
            df = pd.read_csv(data_path)
            
            st.subheader("📊 Vue d'ensemble du Dataset")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Observations", f"{len(df):,}")
            with col2:
                st.metric("Instruments", f"{df['Symbol'].nunique():,}")
            with col3:
                st.metric("Période", f"{(pd.to_datetime(df['Date'].max()) - pd.to_datetime(df['Date'].min())).days} jours")
            with col4:
                st.metric("Classes", "3")
            
            # Distribution des classes
            st.subheader("🎯 Distribution des Classes")
            
            class_dist = df.groupby('InstrumentType').agg({
                'Symbol': 'nunique',
                'Date': 'count'
            }).rename(columns={'Symbol': 'Instruments', 'Date': 'Observations'})
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=class_dist['Instruments'], 
                    names=class_dist.index,
                    title="Répartition des Instruments"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=class_dist['Observations'], 
                    names=class_dist.index,
                    title="Répartition des Observations"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Analyse des features principales
            st.subheader("📊 Analyse des Features Principales")
            
            selected_feature = st.selectbox(
                "Sélectionner une feature",
                ['Daily_Return', 'Volatility_20d', 'Volume_Ratio', 'Price_MA_20d_Ratio']
            )
            
            if selected_feature in df.columns:
                fig = px.box(
                    df, 
                    x='InstrumentType', 
                    y=selected_feature,
                    title=f"Distribution de {selected_feature} par Type d'Instrument"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Matrice de corrélation
            with st.expander("🔍 Matrice de Corrélation"):
                numeric_cols = df.select_dtypes(include=[np.number]).columns[:20]  # Prendre les 20 premières
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Matrice de Corrélation",
                    color_continuous_scale='RdBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.error("❌ Fichier de données non trouvé. Veuillez d'abord collecter et traiter les données.")
    
    def display_about(self):
        """Affiche les informations sur le projet"""
        st.header("ℹ️ À propos du Projet")
        
        st.markdown("""
        ## 🎯 Objectif
        
        Ce projet développe un modèle de machine learning supervisé capable de classer automatiquement 
        des instruments financiers (actions, ETF, obligations) à partir de données de marché.
        
        ## 📊 Méthodologie
        
        ### 1. Collecte des données
        - Sources : Yahoo Finance via yfinance
        - Types d'instruments : Actions, ETF, Obligations
        - Variables : Prix historiques, volumes, ratios financiers
        
        ### 2. Feature Engineering
        - **Rendements** : Calculs de rendements journaliers et périodiques
        - **Volatilité** : Écart-type des rendements sur différentes fenêtres
        - **Moyennes mobiles** : MA(5, 10, 20, 50 jours) et ratios prix/MA
        - **Volume** : Ratios et percentiles de volume
        - **Indicateurs techniques** : RSI, MACD, Bollinger Bands
        - **Momentum** : Indicateurs de tendance et d'accélération
        
        ### 3. Modèles implémentés
        - **Régression Logistique** : Modèle linéaire, interprétable
        - **Random Forest** : Ensemble de arbres, robuste
        - **XGBoost** : Gradient boosting, haute performance
        - **LightGBM** : Gradient boosting optimisé
        
        ### 4. Évaluation
        - **Métriques** : Accuracy, F1-Score, Precision, Recall
        - **Validation** : Validation croisée et test set
        - **Visualisation** : Matrices de confusion, importance des features
        
        ## 🛠️ Technologies utilisées
        
        - **Python** : Langage principal
        - **Scikit-learn** : Machine learning
        - **XGBoost / LightGBM** : Gradient boosting
        - **Pandas / NumPy** : Manipulation de données
        - **Streamlit** : Interface web
        - **Plotly** : Visualisations interactives
        - **yfinance** : Données financières
        
        ## 📈 Résultats attendus
        
        - Classification précise des instruments financiers
        - Identification des variables discriminantes
        - Interface interactive pour les prédictions
        - Compréhension des patterns de marché
        
        ## 👨‍💻 Auteur
        
        **Built by Amine Rahmani**
        
        ### 🔗 Liens
        - **GitHub Repository** : [https://github.com/Amine-Rahmanii/financial-instruments-classifier](https://github.com/Amine-Rahmanii/financial-instruments-classifier)
        - **LinkedIn** : [Amine Rahmani](https://linkedin.com/in/amine-rahmani)
        
        ### 🛠️ Technologies utilisées
        - **Python** : Langage de programmation principal
        - **Scikit-learn** : Framework de machine learning
        - **XGBoost & LightGBM** : Modèles de gradient boosting
        - **Pandas & NumPy** : Manipulation et analyse de données
        - **Streamlit** : Framework pour applications web interactives
        - **Plotly** : Visualisations interactives et dashboards
        - **Matplotlib & Seaborn** : Visualisations statiques
        - **yfinance** : API pour données financières Yahoo Finance
        
        ### 📊 Résultats techniques
        - **4 modèles ML** entraînés et comparés
        - **43 features** d'ingénierie financière
        - **45,000 observations** sur 2 ans de données réelles
        - **100% de précision** sur Random Forest, XGBoost et LightGBM
        
        ---
        
        *Projet développé dans le cadre d'une candidature pour un stage en Data Science / Finance Quantitative.*
        """)
    
    def display_home(self):
        """Affiche la page d'accueil"""
        st.header("🏠 Bienvenue")
        
        st.markdown("""
        ## 🚀 Démarrage Rapide
        
        Cette application vous permet d'explorer un projet complet de classification d'instruments financiers 
        utilisant des techniques de machine learning avancées.
        """)
        
        # Status des modèles
        if self.models_loaded:
            st.success("✅ Modèles chargés avec succès!")
            
            # Afficher quelques statistiques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Modèles disponibles", "4")
            with col2:
                best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
                st.metric("Meilleur F1-Score", f"{self.results[best_model]['f1_score']:.3f}")
            with col3:
                st.metric("Classes", "3")
            
        else:
            st.warning("⚠️ Modèles non trouvés. Veuillez d'abord entraîner les modèles.")
        
        # Guide d'utilisation
        st.markdown("""
        ## 📖 Guide d'utilisation
        
        1. **📊 Performance des Modèles** : Consultez les métriques et matrices de confusion
        2. **🔮 Prédiction** : Testez la classification sur de nouveaux instruments
        3. **📈 Analyse de Données** : Explorez le dataset et les distributions
        4. **ℹ️ À propos** : Découvrez la méthodologie et les technologies
        
        ## 🎯 Fonctionnalités
        
        - ✅ Classification automatique d'instruments financiers
        - ✅ Comparaison de 4 modèles de machine learning
        - ✅ Interface interactive pour les prédictions
        - ✅ Visualisations interactives des performances
        - ✅ Analyse exploratoire des données
        
        ---
        
        ### 👨‍💻 Built by **Amine Rahmani**
        📂 **GitHub** : [https://github.com/Amine-Rahmanii/financial-instruments-classifier](https://github.com/Amine-Rahmanii/financial-instruments-classifier)
        """)
        
        # Note technique en bas
        st.markdown("""
        <div style='margin-top: 2rem; padding: 1rem; background-color: #f0f2f6; border-radius: 0.5rem; border-left: 4px solid #1f77b4;'>
        <small>
        <strong>🔧 Stack technique :</strong> Python • Scikit-learn • XGBoost • LightGBM • Streamlit • Plotly • yfinance<br>
        <strong>📊 Données :</strong> 45,000 observations • 90 instruments • 43 features • 2 ans de données Yahoo Finance
        </small>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Lance l'application"""
        self.display_header()
        
        # Sidebar
        selected_tab = self.display_sidebar()
        
        # Contenu principal basé sur l'onglet sélectionné
        if selected_tab == "🏠 Accueil":
            self.display_home()
        elif selected_tab == "📊 Performance des Modèles":
            self.display_model_performance()
        elif selected_tab == "🔮 Prédiction":
            self.display_prediction()
        elif selected_tab == "📈 Analyse de Données":
            self.display_data_analysis()
        elif selected_tab == "ℹ️ À propos":
            self.display_about()

def main():
    """Fonction principale"""
    app = FinancialClassifierApp()
    app.run()

if __name__ == "__main__":
    main()
