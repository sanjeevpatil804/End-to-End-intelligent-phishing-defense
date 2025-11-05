import pandas as pd
import numpy as np
from networksecurity.Exception.exception import NetworkSecurityException
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifacts
import os
import sys
import pymongo
import certifi
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URI=os.getenv("MONGO_DB_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    def export_data_as_DataFrame(self):
        """
        Read Data from Mongo DB and export as DataFrame"""
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            import certifi
            
            # Configure MongoDB client with correct SSL settings
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URI,
                tls=True,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=5000
            )
            
            # Test connection before proceeding
            try:
                self.mongo_client.admin.command('ping')
            except Exception as e:
                print(f"Failed to connect to MongoDB: {str(e)}")
                raise
                
            collection = self.mongo_client[database_name][collection_name]
            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"], axis=1)
            df.replace(to_replace="na", value=np.nan , inplace=True)
            return df
        except Exception as e:
            raise NetworkSecurityException(e, sys)
    
    def export_data_into_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            dir_path =os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def slit_data_as_train_test(self,dataframe:pd.DataFrame)->None:
        try:
            train_set,test_set=train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )
            train_file_path=self.data_ingestion_config.train_file_path  
            test_file_path=self.data_ingestion_config.test_file_path

            dir_path=os.path.dirname(train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_set.to_csv(train_file_path,index=False,header=True)
            test_set.to_csv(test_file_path,index=False,header=True)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_data_as_DataFrame()
            dataframe=self.export_data_into_feature_store(dataframe=dataframe)
            self.slit_data_as_train_test(dataframe=dataframe)
            dataingesstionartifacts=DataIngestionArtifacts(train_file_path=self.data_ingestion_config.train_file_path,
                                                          test_file_path=self.data_ingestion_config.test_file_path)
            return dataingesstionartifacts
        except Exception as e:
            raise NetworkSecurityException(e, sys)  

   