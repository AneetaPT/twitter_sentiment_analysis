import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import Word2VecProcessor
from src.components.data_transformation import Word2VecProcessorConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        # Allow memory growth to avoid allocating all GPU memory at once
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU configured successfully")
    except RuntimeError as e:
        print(e)
@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/twitter_sentiment.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.raw_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    df=obj.initiate_data_ingestion()

    data_transformation=Word2VecProcessor()
    x_train_arr,x_test_arr,y_train_arr,y_test_arr=data_transformation.initiate_data_transformation(df)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(x_train_arr,x_test_arr,y_train_arr,y_test_arr))


