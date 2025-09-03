"""
Module de préparation des données et feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class FinancialFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_returns(self, df):
        """Calcule les rendements journaliers"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
        return df
    
    def calculate_volatility(self, df, windows=[5, 10, 20]):
        """Calcule la volatilité sur différentes fenêtres"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        for window in windows:
            df[f'Volatility_{window}d'] = df.groupby('Symbol')['Daily_Return'].rolling(
                window=window, min_periods=1
            ).std().reset_index(0, drop=True)
        
        return df
    
    def calculate_moving_averages(self, df, windows=[5, 10, 20, 50]):
        """Calcule les moyennes mobiles"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        for window in windows:
            df[f'MA_{window}d'] = df.groupby('Symbol')['Close'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            # Ratio prix/moyenne mobile
            df[f'Price_MA_{window}d_Ratio'] = df['Close'] / df[f'MA_{window}d']
        
        return df
    
    def calculate_volume_features(self, df):
        """Calcule les features basées sur le volume"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        # Moyenne mobile du volume
        df['Volume_MA_20d'] = df.groupby('Symbol')['Volume'].rolling(
            window=20, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Ratio volume/moyenne
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20d']
        
        # Volume relatif (percentile)
        df['Volume_Percentile'] = df.groupby('Symbol')['Volume'].rank(pct=True)
        
        return df
    
    def calculate_price_features(self, df):
        """Calcule les features basées sur les prix"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        # Range journalier
        df['Daily_Range'] = (df['High'] - df['Low']) / df['Open']
        
        # Position dans le range
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['Close_Position'] = df['Close_Position'].fillna(0.5)
        
        # Gap par rapport au jour précédent
        df['Gap'] = df.groupby('Symbol')['Open'].pct_change()
        
        # Body et shadow des chandelles
        df['Body'] = abs(df['Close'] - df['Open']) / df['Open']
        df['Upper_Shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Open']
        df['Lower_Shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Open']
        
        return df
    
    def calculate_technical_indicators(self, df):
        """Calcule des indicateurs techniques"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        # RSI (Relative Strength Index)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = df.groupby('Symbol')['Close'].apply(
            lambda x: calculate_rsi(x)
        ).reset_index(0, drop=True)
        
        # MACD
        df['EMA_12'] = df.groupby('Symbol')['Close'].ewm(span=12).mean().reset_index(0, drop=True)
        df['EMA_26'] = df.groupby('Symbol')['Close'].ewm(span=26).mean().reset_index(0, drop=True)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df.groupby('Symbol')['MACD'].ewm(span=9).mean().reset_index(0, drop=True)
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        df['BB_Middle'] = df.groupby('Symbol')['Close'].rolling(window=20).mean().reset_index(0, drop=True)
        df['BB_Std'] = df.groupby('Symbol')['Close'].rolling(window=20).std().reset_index(0, drop=True)
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def calculate_momentum_features(self, df):
        """Calcule les features de momentum"""
        df = df.copy()
        df = df.sort_values(['Symbol', 'Date'])
        
        # Momentum sur différentes périodes
        for period in [5, 10, 20]:
            df[f'Momentum_{period}d'] = df.groupby('Symbol')['Close'].pct_change(periods=period)
        
        # Accélération (dérivée seconde)
        df['Acceleration_5d'] = df.groupby('Symbol')['Daily_Return'].diff(periods=5)
        
        return df
    
    def add_time_features(self, df):
        """Ajoute des features temporelles"""
        df = df.copy()
        
        # Convertir la colonne Date en datetime avec UTC puis supprimer le timezone
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        df['DaysFromStart'] = (df['Date'] - df['Date'].min()).dt.days
        
        return df
    
    def prepare_features(self, df, metrics_df=None):
        """
        Prépare toutes les features pour le machine learning
        
        Args:
            df (pd.DataFrame): Données de prix historiques
            metrics_df (pd.DataFrame): Métriques financières supplémentaires
            
        Returns:
            pd.DataFrame: DataFrame avec toutes les features
        """
        print("🔧 Feature Engineering en cours...")
        
        # Calculs des features
        df = self.calculate_returns(df)
        print("  ✅ Rendements calculés")
        
        df = self.calculate_volatility(df)
        print("  ✅ Volatilité calculée")
        
        df = self.calculate_moving_averages(df)
        print("  ✅ Moyennes mobiles calculées")
        
        df = self.calculate_volume_features(df)
        print("  ✅ Features de volume calculées")
        
        df = self.calculate_price_features(df)
        print("  ✅ Features de prix calculées")
        
        df = self.calculate_technical_indicators(df)
        print("  ✅ Indicateurs techniques calculés")
        
        df = self.calculate_momentum_features(df)
        print("  ✅ Features de momentum calculées")
        
        df = self.add_time_features(df)
        print("  ✅ Features temporelles ajoutées")
        
        # Ajouter les métriques financières si disponibles
        if metrics_df is not None:
            df = df.merge(metrics_df, on='Symbol', how='left')
            print("  ✅ Métriques financières ajoutées")
        
        # Nettoyer les données
        df = self.clean_data(df)
        print("  ✅ Données nettoyées")
        
        return df
    
    def clean_data(self, df):
        """Nettoie les données"""
        df = df.copy()
        
        # Supprimer les lignes avec trop de valeurs manquantes
        df = df.dropna(subset=['Close', 'Volume', 'Daily_Return'])
        
        # Remplacer les infinités par NaN puis par la médiane
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Remplir les NaN avec la médiane pour les features numériques
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col not in ['Label', 'DayOfWeek', 'Month', 'Quarter']]
        
        for col in numeric_columns:
            if df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        # Supprimer les outliers extrêmes (au-delà de 3 écarts-types)
        for col in ['Daily_Return', 'Volume_Ratio']:
            if col in df.columns:
                q99 = df[col].quantile(0.99)
                q01 = df[col].quantile(0.01)
                df[col] = df[col].clip(lower=q01, upper=q99)
        
        return df
    
    def prepare_for_ml(self, df):
        """
        Prépare les données pour le machine learning
        
        Args:
            df (pd.DataFrame): DataFrame avec les features
            
        Returns:
            tuple: (X, y, feature_names)
        """
        # Colonnes à exclure des features
        exclude_columns = [
            'Date', 'Symbol', 'InstrumentType', 'Label',
            'Open', 'High', 'Low', 'Close', 'Volume',  # Prix bruts
            'EMA_12', 'EMA_26', 'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower',  # Intermediaires
            'Volume_MA_20d'  # Intermediaires
        ]
        
        # Sélectionner les features
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        X = df[feature_columns].copy()
        y = df['Label'].copy()
        
        # Gérer les valeurs manquantes restantes
        X = X.fillna(X.median())
        
        print(f"📊 Dataset préparé: {X.shape[0]} observations, {X.shape[1]} features")
        print(f"📊 Distribution des classes: {y.value_counts().to_dict()}")
        
        return X, y, feature_columns

def main():
    """Fonction principale pour préparer les données"""
    
    # Charger les données
    print("📂 Chargement des données...")
    df = pd.read_csv('data/raw_financial_data.csv')
    
    try:
        metrics_df = pd.read_csv('data/financial_metrics.csv')
    except FileNotFoundError:
        metrics_df = None
        print("⚠️ Fichier des métriques non trouvé, continuons sans")
    
    # Préparer les features
    engineer = FinancialFeatureEngineer()
    df_processed = engineer.prepare_features(df, metrics_df)
    
    # Sauvegarder les données préparées
    df_processed.to_csv('data/processed_financial_data.csv', index=False)
    print("💾 Données préparées sauvegardées dans 'data/processed_financial_data.csv'")
    
    # Préparer pour le ML
    X, y, feature_names = engineer.prepare_for_ml(df_processed)
    
    # Sauvegarder les datasets ML
    X.to_csv('data/X_features.csv', index=False)
    pd.DataFrame(y).to_csv('data/y_labels.csv', index=False)
    pd.DataFrame(feature_names, columns=['feature']).to_csv('data/feature_names.csv', index=False)
    
    print("💾 Datasets ML sauvegardés")
    print("✅ Préparation des données terminée!")

if __name__ == "__main__":
    main()
