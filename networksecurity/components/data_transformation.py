import os , sys
from networksecurity.Exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.entity.artifact_entity import DataValidationArtifacts, DataTransformationArtifacts
import pandas as pd
import numpy as np
from networksecurity.Utils.main_utils.utils import save_numpy_array_data, save_object
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constants.training_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS



class DataTransformation:
    def __init__(self, data_validation_artifacts: DataValidationArtifacts,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifacts = data_validation_artifacts
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
      
    @staticmethod
    def read_data(file_path):
        return pd.read_csv(file_path)
    
    def data_transform_object(cls):
        try:
            knn_imputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            imputer_pipeline=Pipeline([("imputer",knn_imputer)])
            return imputer_pipeline
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
      
    def initiate_data_transformation(self)->DataTransformationArtifacts:
        try:
            train_df=DataTransformation.read_data(self.data_validation_artifacts.valid_train_file_path)
            test_df=DataTransformation.read_data(self.data_validation_artifacts.valid_test_file_path)

            input_feature_train_df=train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            target_feature_train_df = target_feature_train_df.replace(-1, 0)

          
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)

            preprocessor=self.data_transform_object()
            preprocessor_object=preprocessor.fit(input_feature_train_df)
            transformed_input_feature_train_df=preprocessor.transform(input_feature_train_df)
            transformed_input_feature_test_df=preprocessor.transform(input_feature_test_df) 

            train_arr=np.c_[transformed_input_feature_train_df,np.array(target_feature_train_df)]
            test_arr=np.c_[transformed_input_feature_test_df,np.array(target_feature_test_df)]  

            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)

            save_object( "final_model/preprocessor.pkl", preprocessor_object,)


            

            data_transformation_artifact=DataTransformationArtifacts( 
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact




        except Exception as e:
            raise NetworkSecurityException(e, sys)
