"""
Module d'entraînement et d'évaluation des modèles de classification
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
        # Mapping des labels vers les noms
        self.label_names = {0: 'Stock', 1: 'ETF', 2: 'Bond'}
        
    def prepare_data(self, X, y, test_size=0.2, random_state=42):
        """Prépare les données pour l'entraînement"""
        
        # Division train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalisation
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
        
    def train_logistic_regression(self, X_train, y_train):
        """Entraîne un modèle de régression logistique"""
        print("🔄 Entraînement Régression Logistique...")
        
        # Grille de paramètres
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'max_iter': [1000, 2000],
            'solver': ['liblinear', 'lbfgs']
        }
        
        # GridSearch avec validation croisée
        lr = LogisticRegression(random_state=42)
        grid_search = GridSearchCV(
            lr, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['logistic_regression'] = grid_search.best_estimator_
        print(f"  ✅ Meilleurs paramètres: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train, y_train):
        """Entraîne un modèle Random Forest"""
        print("🔄 Entraînement Random Forest...")
        
        # Grille de paramètres
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        
        # GridSearch avec validation croisée
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['random_forest'] = grid_search.best_estimator_
        print(f"  ✅ Meilleurs paramètres: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train, y_train):
        """Entraîne un modèle XGBoost"""
        print("🔄 Entraînement XGBoost...")
        
        # Grille de paramètres
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        # GridSearch avec validation croisée
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['xgboost'] = grid_search.best_estimator_
        print(f"  ✅ Meilleurs paramètres: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def train_lightgbm(self, X_train, y_train):
        """Entraîne un modèle LightGBM"""
        print("🔄 Entraînement LightGBM...")
        
        # Grille de paramètres
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100]
        }
        
        # GridSearch avec validation croisée
        lgb_model = lgb.LGBMClassifier(random_state=42, verbose=-1)
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        self.models['lightgbm'] = grid_search.best_estimator_
        print(f"  ✅ Meilleurs paramètres: {grid_search.best_params_}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Évalue un modèle"""
        
        # Prédictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Métriques
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        
        # Rapport de classification
        class_report = classification_report(
            y_test, y_pred, 
            target_names=[self.label_names[i] for i in sorted(self.label_names.keys())],
            output_dict=True
        )
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Sauvegarder les résultats
        self.results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"📊 {model_name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        return self.results[model_name]
    
    def get_feature_importance(self, model, feature_names, model_name, top_n=20):
        """Récupère l'importance des features"""
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
        else:
            return None
        
        # Créer un DataFrame avec les importances
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def plot_confusion_matrices(self, save_path='models/confusion_matrices.png'):
        """Génère les matrices de confusion pour tous les modèles"""
        
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        class_names = [self.label_names[i] for i in sorted(self.label_names.keys())]
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            if idx < len(axes):
                sns.heatmap(
                    results['confusion_matrix'], 
                    annot=True, 
                    fmt='d',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    ax=axes[idx],
                    cmap='Blues'
                )
                axes[idx].set_title(f'{model_name.replace("_", " ").title()}')
                axes[idx].set_xlabel('Prédictions')
                axes[idx].set_ylabel('Vraies valeurs')
        
        # Supprimer les axes vides
        for idx in range(len(self.results), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Matrices de confusion sauvegardées: {save_path}")
    
    def plot_feature_importance(self, model, feature_names, model_name, save_path=None):
        """Génère un graphique d'importance des features"""
        
        importance_df = self.get_feature_importance(model, feature_names, model_name)
        
        if importance_df is not None:
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df.head(15), x='importance', y='feature')
            plt.title(f'Top 15 Features - {model_name.replace("_", " ").title()}')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            if save_path is None:
                save_path = f'models/feature_importance_{model_name}.png'
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Importance des features sauvegardée: {save_path}")
            
            return importance_df
    
    def train_all_models(self, X, y, feature_names):
        """Entraîne tous les modèles"""
        
        print("🚀 Début de l'entraînement de tous les modèles...")
        
        # Préparer les données
        X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = self.prepare_data(X, y)
        
        # Entraîner les modèles
        self.train_logistic_regression(X_train_scaled, y_train)
        self.train_random_forest(X_train, y_train)  # Random Forest ne nécessite pas de normalisation
        self.train_xgboost(X_train, y_train)
        self.train_lightgbm(X_train, y_train)
        
        print("\n📊 Évaluation des modèles...")
        
        # Évaluer les modèles
        self.evaluate_model(self.models['logistic_regression'], X_test_scaled, y_test, 'logistic_regression')
        self.evaluate_model(self.models['random_forest'], X_test, y_test, 'random_forest')
        self.evaluate_model(self.models['xgboost'], X_test, y_test, 'xgboost')
        self.evaluate_model(self.models['lightgbm'], X_test, y_test, 'lightgbm')
        
        # Générer les visualisations
        self.plot_confusion_matrices()
        
        # Feature importance pour les modèles tree-based
        for model_name in ['random_forest', 'xgboost', 'lightgbm']:
            self.plot_feature_importance(self.models[model_name], feature_names, model_name)
        
        # Sauvegarder les modèles
        self.save_models()
        
        # Afficher le résumé
        self.print_summary()
        
        return self.models, self.results
    
    def save_models(self):
        """Sauvegarde tous les modèles"""
        
        for model_name, model in self.models.items():
            model_path = f'models/{model_name}.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Sauvegarder le scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Sauvegarder les résultats
        with open('models/results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        print("💾 Modèles sauvegardés dans le dossier 'models/'")
    
    def print_summary(self):
        """Affiche un résumé des performances"""
        
        print("\n" + "="*60)
        print("📊 RÉSUMÉ DES PERFORMANCES")
        print("="*60)
        
        # Créer un DataFrame de résumé
        summary_data = []
        for model_name, results in self.results.items():
            summary_data.append({
                'Modèle': model_name.replace('_', ' ').title(),
                'Accuracy': f"{results['accuracy']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Meilleur modèle
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        print(f"\n🏆 Meilleur modèle: {best_model.replace('_', ' ').title()}")
        print(f"   F1-Score: {self.results[best_model]['f1_score']:.4f}")

def main():
    """Fonction principale pour entraîner les modèles"""
    
    print("📂 Chargement des données...")
    
    # Charger les données
    X = pd.read_csv('data/X_features.csv')
    y = pd.read_csv('data/y_labels.csv').values.ravel()
    feature_names = pd.read_csv('data/feature_names.csv')['feature'].tolist()
    
    print(f"✅ Données chargées: {X.shape[0]} observations, {X.shape[1]} features")
    
    # Entraîner les modèles
    trainer = ModelTrainer()
    models, results = trainer.train_all_models(X, y, feature_names)
    
    print("\n✅ Entraînement terminé!")

if __name__ == "__main__":
    main()
