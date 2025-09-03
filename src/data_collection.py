"""
Module de collecte de données financières
Récupère des données pour différents types d'instruments financiers
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialDataCollector:
    def __init__(self):
        # Dictionnaire des symboles par type d'instrument
        self.instruments = {
            'stock': {
                'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V',
                           'PG', 'UNH', 'HD', 'MA', 'BAC', 'ABBV', 'PFE', 'KO', 'AVGO', 'PEP',
                           'TMO', 'COST', 'DHR', 'ABT', 'VZ', 'ADBE', 'NFLX', 'CRM', 'TXN', 'XOM'],
                'label': 0
            },
            'etf': {
                'symbols': ['SPY', 'QQQ', 'VTI', 'IWM', 'EFA', 'VEA', 'VWO', 'VNQ', 'GLD', 'SLV',
                           'TLT', 'IEF', 'LQD', 'HYG', 'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP',
                           'XLU', 'XLY', 'XLB', 'XLRE', 'VGT', 'VHT', 'VFH', 'VIS', 'VDE', 'VAW'],
                'label': 1
            },
            'bond': {
                'symbols': ['AGG', 'BND', 'VTEB', 'MUB', 'VCIT', 'VCSH', 'VGIT', 'VGSH', 'SCHZ', 'GOVT',
                           'SHY', 'IEI', 'TIP', 'VTIP', 'SCHP', 'EMB', 'BWX', 'BNDX', 'VXUS', 'IAUM',
                           'JPST', 'NEAR', 'MINT', 'ICSH', 'FLOT', 'USFR', 'TFLO', 'SCHO', 'SCHR', 'SPTS'],
                'label': 2
            }
        }
        
    def collect_data(self, period="2y", interval="1d"):
        """
        Collecte les données pour tous les instruments
        
        Args:
            period (str): Période de données ('1y', '2y', '5y', etc.)
            interval (str): Intervalle ('1d', '1wk', etc.)
            
        Returns:
            pd.DataFrame: DataFrame consolidé avec toutes les données
        """
        all_data = []
        
        print("🔄 Collecte des données en cours...")
        
        for instrument_type, config in self.instruments.items():
            print(f"  📊 Collecte {instrument_type}...")
            
            for symbol in config['symbols']:
                try:
                    # Télécharger les données
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=period, interval=interval)
                    
                    if len(data) > 0:
                        # Ajouter les métadonnées
                        data['Symbol'] = symbol
                        data['InstrumentType'] = instrument_type
                        data['Label'] = config['label']
                        data.reset_index(inplace=True)
                        
                        all_data.append(data)
                        
                except Exception as e:
                    print(f"    ⚠️ Erreur pour {symbol}: {e}")
                    continue
        
        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            print(f"✅ Collecte terminée: {len(df)} observations pour {df['Symbol'].nunique()} instruments")
            return df
        else:
            raise Exception("Aucune donnée collectée")
    
    def get_additional_metrics(self, symbols):
        """
        Récupère des métriques financières supplémentaires
        
        Args:
            symbols (list): Liste des symboles
            
        Returns:
            pd.DataFrame: DataFrame avec les métriques
        """
        metrics_data = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                metrics = {
                    'Symbol': symbol,
                    'MarketCap': info.get('marketCap', np.nan),
                    'Beta': info.get('beta', np.nan),
                    'PE_Ratio': info.get('trailingPE', np.nan),
                    'DividendYield': info.get('dividendYield', np.nan),
                    'Volume_Avg': info.get('averageVolume', np.nan),
                    'Price52WeekHigh': info.get('fiftyTwoWeekHigh', np.nan),
                    'Price52WeekLow': info.get('fiftyTwoWeekLow', np.nan)
                }
                
                metrics_data.append(metrics)
                
            except Exception as e:
                print(f"    ⚠️ Erreur métriques pour {symbol}: {e}")
                continue
        
        return pd.DataFrame(metrics_data)

def main():
    """Fonction principale pour collecter et sauvegarder les données"""
    
    collector = FinancialDataCollector()
    
    # Collecter les données historiques
    df = collector.collect_data(period="2y", interval="1d")
    
    # Sauvegarder les données brutes
    df.to_csv('data/raw_financial_data.csv', index=False)
    print("💾 Données sauvegardées dans 'data/raw_financial_data.csv'")
    
    # Collecter les métriques supplémentaires
    all_symbols = df['Symbol'].unique()
    metrics_df = collector.get_additional_metrics(all_symbols)
    metrics_df.to_csv('data/financial_metrics.csv', index=False)
    print("💾 Métriques sauvegardées dans 'data/financial_metrics.csv'")
    
    # Afficher un résumé
    print("\n📈 Résumé des données collectées:")
    print(f"Total observations: {len(df):,}")
    print(f"Période: {df['Date'].min().strftime('%Y-%m-%d')} à {df['Date'].max().strftime('%Y-%m-%d')}")
    print("\nRépartition par type d'instrument:")
    print(df.groupby('InstrumentType').agg({
        'Symbol': 'nunique',
        'Date': 'count'
    }).rename(columns={'Symbol': 'Nb_Instruments', 'Date': 'Nb_Observations'}))

if __name__ == "__main__":
    main()
