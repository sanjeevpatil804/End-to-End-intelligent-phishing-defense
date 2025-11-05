
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.Data_validation import DataValidation

from networksecurity.Exception.exception import NetworkSecurityException
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig, ModelTrainerConfig, TrainingPipelineConfig
import sys



if __name__=='__main__':
    try:
        training_pipeline_config=TrainingPipelineConfig()
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion_artifacts=data_ingestion.initiate_data_ingestion()
        
        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifacts,data_validation_config)
        data_validation_artifacts=data_validation.initiate_data_validation()


        data_transformation_config=DataTransformationConfig(training_pipeline_config)
        data_transformation=DataTransformation(data_validation_artifacts, data_transformation_config)
        data_transformation_artifacts=data_transformation.initiate_data_transformation()
        
        model_trainer_config=ModelTrainerConfig(training_pipeline_config)
        model_trainer=ModelTrainer(model_trainer_config=model_trainer_config,data_transformation_artifact=data_transformation_artifacts)
        model_trainer_artifact=model_trainer.initiate_model_trainer()



    except Exception as e:
        raise NetworkSecurityException(e,sys)