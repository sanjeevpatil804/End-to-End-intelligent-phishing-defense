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
from datetime import datetime


class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifacts):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def train_model(self,X_train,y_train,x_test,y_test):
        try:
      
            # Optimize models using Optuna
            optimization_results = optimize_models(
                X_train=X_train,
                y_train=y_train,
                n_trials=50,
                classifier_types=['rf', 'xgb'],
                study_name='phishing_detection_study',
                verbose=True
            )
            
            # Extract best model and results
            best_model = optimization_results['best_model']
            best_params = optimization_results['best_params']
            best_score = optimization_results['best_score']
            best_classifier = optimization_results['best_classifier']
            
            # Create model report
            model_report = {
                "best_model": best_classifier,
                "best_cv_f1_score": float(best_score),
                "n_trials": optimization_results['all_trials'],
                "best_params": {k: str(v) for k, v in best_params.items()}
            }
      
            
            # Evaluate on training data
            y_train_pred = best_model.predict(X_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            model_report['train_f1_score'] = float(classification_train_metric.f1_score)
            model_report['train_precision'] = float(classification_train_metric.precision_score)
            model_report['train_recall'] = float(classification_train_metric.recall_score)
            
            print(f"📊 Training Metrics:")
            print(f"   F1 Score:    {classification_train_metric.f1_score:.4f}")
            print(f"   Precision:   {classification_train_metric.precision_score:.4f}")
            print(f"   Recall:      {classification_train_metric.recall_score:.4f}")
            
            # Evaluate on test data
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            
            model_report['test_f1_score'] = float(classification_test_metric.f1_score)
            model_report['test_precision'] = float(classification_test_metric.precision_score)
            model_report['test_recall'] = float(classification_test_metric.recall_score)
            
            print(f"\n📊 Test Metrics:")
            print(f"   F1 Score:    {classification_test_metric.f1_score:.4f}")
            print(f"   Precision:   {classification_test_metric.precision_score:.4f}")
            print(f"   Recall:      {classification_test_metric.recall_score:.4f}")
            
            # Save model evaluation report
            model_report_file_path = os.path.join(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                "model_evaluation_report.yaml"
            )
            write_yaml_file(file_path=model_report_file_path, content=model_report)
            
            print(f"\n💾 Model report saved: {model_report_file_path}")
            
            # Load preprocessor and save complete pipeline
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            # Create NetworkModel instance (combining preprocessor + model)
            Network_Model_obj = NetworkModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=Network_Model_obj)
            
            # Save final model separately
            os.makedirs("final_model", exist_ok=True)
            save_object("final_model/model.pkl", best_model)
            
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
            print(f"Best Model: {best_classifier.upper()}")
            print(f"Test F1 Score: {classification_test_metric.f1_score:.4f}")
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