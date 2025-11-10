import optuna
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from networksecurity.Exception.exception import NetworkSecurityException
import sys
import os


def objective(trial, X_train, y_train, classifier_types):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training labels
        classifier_types: List of classifier types to try ['rf', 'xgb']
    
    Returns:
        float: Mean F1 score from cross-validation
    """
    try:
        classifier_name = trial.suggest_categorical('classifier', classifier_types)
    
        if classifier_name == 'rf':
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
            
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                criterion=criterion,
                max_features=max_features,
                random_state=42,
                n_jobs=-1
            )

        elif classifier_name == 'xgb':
            n_estimators = trial.suggest_int('n_estimators', 100, 500)
            max_depth = trial.suggest_int('max_depth', 3, 10)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            subsample = trial.suggest_float('subsample', 0.6, 1.0)
            colsample_bytree = trial.suggest_float('colsample_bytree', 0.6, 1.0)
            reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)

            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                tree_method='hist',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier_name}")
        
        # Stratified K-Fold cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        mean_score = scores.mean()
        
        return mean_score
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)



def optimize_models(X_train, y_train, n_trials=50, classifier_types=['rf', 'xgb'], 
                   study_name='phishing_detection_study', verbose=True):
  
    try:
       
        # Create study
        study =optuna.create_study(study_name=study_name, direction='maximize')
        
        # Optimize
        study.optimize(
            lambda trial: objective(trial, X_train, y_train, classifier_types),
            n_trials=n_trials,
            show_progress_bar=verbose
        )
        
        # Extract best results
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        best_classifier = best_params['classifier']
        
        
        # Train final model with best parameters
        if best_classifier == 'rf':
            best_model = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                criterion=best_params['criterion'],
                max_features=best_params['max_features'],
                random_state=42,
                n_jobs=-1
            )
        elif best_classifier == 'xgb':
            best_model = XGBClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                learning_rate=best_params['learning_rate'],
                subsample=best_params['subsample'],
                colsample_bytree=best_params['colsample_bytree'],
                reg_alpha=best_params['reg_alpha'],
                reg_lambda=best_params['reg_lambda'],
                tree_method='hist',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown classifier: {best_classifier}")
        
        # Fit on full training data
        best_model.fit(X_train, y_train)
        
        return {
            'study': study,
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'best_classifier': best_classifier,
            'all_trials': len(study.trials)
        }
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)
  
    
