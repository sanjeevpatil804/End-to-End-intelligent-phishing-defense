import os
import sys
from networksecurity.Exception.exception import NetworkSecurityException 
from networksecurity.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.Utils.main_utils.estimator  import NetworkModel
from networksecurity.Utils.main_utils.utils import save_object,load_object
from networksecurity.Utils.main_utils.utils import load_numpy_array_data,write_yaml_file
from networksecurity.Utils.main_utils.classification_metrics import get_classification_score
from networksecurity.Utils.main_utils.optuna_tuner import optimize_models
import mlflow
import mlflow.sklearn
from datetime import datetime





class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifacts):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
            
            # Configure MLflow tracking for real-time monitoring
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("Network_Security_Model_Training")
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def train_model(self,X_train,y_train,x_test,y_test):
        try:
            # Start a single MLflow run for the entire training session
            with mlflow.start_run(run_name=f"Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                
                # Log dataset information
                mlflow.log_param("train_samples", X_train.shape[0])
                mlflow.log_param("test_samples", x_test.shape[0])
                mlflow.log_param("n_features", X_train.shape[1])
                mlflow.log_param("optimization_method", "Optuna")
                
                print(f"\n{'='*70}")
                print("🚀 STARTING MODEL TRAINING WITH OPTUNA OPTIMIZATION")
                print(f"{'='*70}")
                print(f"Training samples: {X_train.shape[0]}")
                print(f"Test samples: {x_test.shape[0]}")
                print(f"Features: {X_train.shape[1]}")
                print(f"{'='*70}\n")
                
                # Optimize models using Optuna
                optimization_results = optimize_models(
                    X_train, y_train, x_test, y_test,
                    models_to_optimize=['rf', 'xgb'],
                    n_trials=50
                )
                
                # Extract results
                rf_result = optimization_results.get('Random Forest', {})
                xgb_result = optimization_results.get('XGBoost', {})
                
                rf_score = rf_result.get('score', 0)
                xgb_score = xgb_result.get('score', 0)
                
                # Create model report
                model_report = {
                    "Random Forest": rf_score,
                    "XGBoost": xgb_score
                }
                
                # Log model comparison scores
                mlflow.log_metric("RandomForest_best_f1", rf_score)
                mlflow.log_metric("XGBoost_best_f1", xgb_score)
                
                # Save model evaluation report
                model_report_file_path = os.path.join(
                    os.path.dirname(self.model_trainer_config.trained_model_file_path),
                    "model_evaluation_report.yaml"
                )
                write_yaml_file(file_path=model_report_file_path, content=model_report)
                mlflow.log_artifact(model_report_file_path)
                
                print(f"\n{'='*70}")
                print("MODEL EVALUATION REPORT:")
                print(f"{'='*70}")
                for model_name, score in model_report.items():
                    print(f"{model_name:25} : {score:.4f}")
                print(f"{'='*70}\n")
                
                # Select best model
                if rf_score > xgb_score:
                    best_model_name = "Random Forest"
                    best_model_score = rf_score
                    best_model = rf_result['model']
                    best_params = rf_result['params']
                else:
                    best_model_name = "XGBoost"
                    best_model_score = xgb_score
                    best_model = xgb_result['model']
                    best_params = xgb_result['params']
                
                print(f"\n🎯 Best Model Selected: {best_model_name}")
                
                # Log best model information
                mlflow.log_param("best_model_name", best_model_name)
                mlflow.log_metric("best_model_f1_score", best_model_score)
                
                # Log best model hyperparameters
                for param_name, param_value in best_params.items():
                    mlflow.log_param(f"best_{param_name}", str(param_value))
                
                # Evaluate on training data
                y_train_pred = best_model.predict(X_train)
                classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
                
                # Log training metrics
                mlflow.log_metric("train_f1_score", classification_train_metric.f1_score)
                mlflow.log_metric("train_precision_score", classification_train_metric.precision_score)
                mlflow.log_metric("train_recall_score", classification_train_metric.recall_score)
                
                print(f"\n📊 Training Metrics:")
                print(f"   F1 Score:    {classification_train_metric.f1_score:.4f}")
                print(f"   Precision:   {classification_train_metric.precision_score:.4f}")
                print(f"   Recall:      {classification_train_metric.recall_score:.4f}")
                
                # Evaluate on test data
                y_test_pred = best_model.predict(x_test)
                classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
                
                # Log test metrics
                mlflow.log_metric("test_f1_score", classification_test_metric.f1_score)
                mlflow.log_metric("test_precision_score", classification_test_metric.precision_score)
                mlflow.log_metric("test_recall_score", classification_test_metric.recall_score)
                
                print(f"\n📊 Test Metrics:")
                print(f"   F1 Score:    {classification_test_metric.f1_score:.4f}")
                print(f"   Precision:   {classification_test_metric.precision_score:.4f}")
                print(f"   Recall:      {classification_test_metric.recall_score:.4f}")
                
                # Log model to MLflow
                mlflow.sklearn.log_model(best_model, "best_model")
                
                # Load preprocessor and save complete pipeline
                preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
                
                model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
                os.makedirs(model_dir_path, exist_ok=True)
                
                # Create NetworkModel instance (combining preprocessor + model)
                Network_Model_obj = NetworkModel(preprocessor=preprocessor, model=best_model)
                save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model_obj)
                
                # Save final model
                save_object("final_model/model.pkl", best_model)
                
                # Log model artifacts
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path)
                
                print(f"\n💾 Models saved successfully!")
                print(f"   Pipeline: {self.model_trainer_config.trained_model_file_path}")
                print(f"   Final Model: final_model/model.pkl")
                
                # Create and return Model Trainer Artifact
                model_trainer_artifact = ModelTrainerArtifact(
                    trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                    train_metric_artifact=classification_train_metric,
                    test_metric_artifact=classification_test_metric
                )
                
                print(f"\n{'='*70}")
                print("✨ TRAINING COMPLETE!")
                print(f"{'='*70}")
                print(f"Best Model: {best_model_name}")
                print(f"Test F1 Score: {classification_test_metric.f1_score:.4f}")
                print(f"MLflow UI: http://localhost:5000")
                print(f"{'='*70}\n")
                
                return model_trainer_artifact
                
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise NetworkSecurityException(e,sys)