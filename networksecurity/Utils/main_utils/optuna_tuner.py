import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import mlflow
from networksecurity.Exception.exception import NetworkSecurityException
import sys


def optimize_random_forest(X_train, y_train, x_test, y_test, n_trials):
    """
    Optimize Random Forest hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        n_trials: Number of optimization trials
        
    Returns:
        tuple: (best_score, best_params, best_model)
    """
    try:
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
                'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.01),
                'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                'random_state': 42,
                'n_jobs': -1,
            }
         

            model = RandomForestClassifier(**params, verbose=1)
            # Use CV on training data only to avoid test leakage
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)
            score = float(scores.mean())

            # Log each trial to MLflow in real-time
            mlflow.log_metric(f"RF_trial_{trial.number}_f1", score, step=trial.number)

            return score
        
        # Optimize with Optuna
        print("🔍 Optimizing Random Forest with Optuna...")
        study = optuna.create_study(direction='maximize', study_name='RandomForest')
        study.optimize(objective_rf, n_trials=n_trials, show_progress_bar=True)
        
        best_score = study.best_value
        best_params = study.best_params
        # Ensure deterministic seed and parallel threads for the final model too
        best_params['random_state'] = 42
        if 'n_jobs' not in best_params:
            best_params['n_jobs'] = -1
        
        # Create best model
        best_model = RandomForestClassifier(**best_params, verbose=1)
        best_model.fit(X_train, y_train)
        
        print(f"\n✅ Random Forest Optimization Complete!")
        print(f"   Best F1 Score: {best_score:.4f}")
        print(f"   Best Parameters: {best_params}")
        
        return best_score, best_params, best_model
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def optimize_xgboost(X_train, y_train, x_test, y_test, n_trials):
    """
    Optimize XGBoost hyperparameters using Optuna
    
    Args:
        X_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        n_trials: Number of optimization trials
        
    Returns:
        tuple: (best_score, best_params, best_model)
    """
    try:
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
                'scale_pos_weight': trial.suggest_categorical('scale_pos_weight', [1.0, 2.0, 5.0, 10.0]),
                'tree_method': 'hist',
                'random_state': 42,
                'eval_metric': 'logloss',
                'objective': 'binary:logistic'
            }

            model = XGBClassifier(**params)
            # Use CV on training data only to avoid test leakage
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)
            score = float(scores.mean())

            # Log each trial to MLflow in real-time
            mlflow.log_metric(f"XGB_trial_{trial.number}_f1", score, step=trial.number)

            return score
        
        # Optimize with Optuna
        print("\n🔍 Optimizing XGBoost with Optuna...")
        study = optuna.create_study(direction='maximize', study_name='XGBoost')
        study.optimize(objective_xgb, n_trials=n_trials, show_progress_bar=True)
        
        best_score = study.best_value
        best_params = study.best_params
        # Keep core deterministic/compat settings
        best_params['random_state'] = 42
        best_params['eval_metric'] = 'logloss'
        best_params['objective'] = 'binary:logistic'
        if 'tree_method' not in best_params:
            best_params['tree_method'] = 'hist'
        
        # Create best model
        best_model = XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)
        
        print(f"\n✅ XGBoost Optimization Complete!")
        print(f"   Best F1 Score: {best_score:.4f}")
        print(f"   Best Parameters: {best_params}")
        
        return best_score, best_params, best_model
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def optimize_models(X_train, y_train, x_test, y_test, models_to_optimize=['rf', 'xgb'], n_trials=50 ):
    """
    Optimize multiple models and return results
    
    Args:
        X_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        models_to_optimize: List of model names ('rf', 'xgb')
        n_trials: Number of optimization trials per model
        
    Returns:
        dict: Dictionary with model results
    """
    try:
        results = {}
        
        if 'rf' in models_to_optimize:
            rf_score, rf_params, rf_model = optimize_random_forest(
                X_train, y_train, x_test, y_test, n_trials=n_trials
            )
            results['Random Forest'] = {
                'score': rf_score,
                'params': rf_params,
                'model': rf_model
            }
        
        if 'xgb' in models_to_optimize:
            xgb_score, xgb_params, xgb_model = optimize_xgboost(
                X_train, y_train, x_test, y_test, n_trials=n_trials
            )
            results['XGBoost'] = {
                'score': xgb_score,
                'params': xgb_params,
                'model': xgb_model
            }
        
        return results
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
