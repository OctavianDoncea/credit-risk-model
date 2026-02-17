import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, accuracy_score)
from imblearn.over_sampling import SMOTE

def load_and_split_data(filepath, test_size=0.2):
    print("Loading modelling data...")
    df = pd.read_csv(filepath)

    X = df.drop('defaulted', axis=1)
    y = df['defaulted']

    print(f'Dataset shape: {X.shape}')
    print(f'Default rate: {y.mean()*100:.2f}%')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    print(f'Train set: {X_train.shape}')
    print(f'Test set: {X_test.shape}')

    return X_train, X_test, y_train, y_test, X.columns.tolist()

def handle_class_imbalance(X_train, y_train, method='smote'):
    if method == 'smote':
        print('Applying SMOTE to balance classes...')
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f'Original class disitribution: {np.bincount(y_train)}')
        print(f'Balanced class distribution: {np.bincount(y_train_balanced)}')

        return X_train_balanced, y_train_balanced

    return X_train, y_train

def train_logistic_regression(X_train, y_train, X_test, y_test):
    print('\n==TRAINING LOGISTIC REGRESSION==')

    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'AUC: {auc:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    return model, y_pred_proba, auc

def train_random_forest(X_train, y_train, X_test, y_test):
    print('\n==TRAINING RANDOM FOREST==')

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'AUC: {auc:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    return model, y_pred_proba, auc

def train_xgboost(X_train, y_train, X_test, y_test):
    print('\n==TRAINING XGBOOST==')

    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    print(f'AUC: {auc:.4f}')
    print(f'Accuracy: {accuracy:.4f}')

    return model, y_pred_proba, auc

def plot_roc_curve(y_test, y_pred_probas, model_names):
    plt.figure(figsize=(10, 8))

    for y_pred_proba, name in zip(y_pred_probas, model_names):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comaprison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('../models/roc_curve.png', dpi=300)

def plot_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14)
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        plt.savefig('../models/feature_importance.png', dpi=300)
        
        return importance_df

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f'Model saved to {filepath}')

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data('../data/processed/modelling_data.csv')
    models = {}
    predictions = {}
    aucs = {}

    lr_model, lr_pred, lr_auc = train_logistic_regression(X_train, y_train, X_test, y_test)
    models['Logistic Regression'] = lr_model
    predictions['Logistic Regression'] = lr_pred
    aucs['Logistic Regression'] = lr_auc

    rf_model, rf_pred, rf_auc = train_random_forest(X_train, y_train, X_test, y_test)
    models['Random Forest'] = rf_model
    predictions['Random Forest'] = rf_pred
    aucs['Random Forest'] = rf_auc

    xgb_model, xgb_pred, xgb_auc = train_xgboost(X_train, y_train, X_test, y_test)
    models['XGBoost'] = xgb_model
    predictions['XGBoost'] = xgb_pred
    aucs['XGBoost'] = xgb_auc

    print('\n==MODEL COMPARISON==')
    for name, auc in aucs.items():
        print(f'{name}: AUC = {auc:.4f}')

    best_model_name = max(aucs, key=aucs.get)
    best_model = models[best_model_name]

    print(f'\nBest Model: {best_model_name} (AUC = {aucs[best_model_name]:.4f})')

    plot_roc_curve(y_test, [predictions[name] for name in models.keys()], list(models.keys()))

    if best_model_name in ['Random Forest', 'XGBoost']:
        importance_df = plot_feature_importance(best_model, feature_names)
        importance_df.to_csv('../models/feature_importance.csv', index=False)

    save_model(best_model, f'../models/{best_model_name.lower().replace(" ", "_")}_model.pkl')

    metadata = {
        'model_name': best_model_name,
        'auc': aucs[best_model_name],
        'feature_names': feature_names,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    joblib.dump(metadata, '../models/model_metadata.pkl')

    print('Model training complete!')