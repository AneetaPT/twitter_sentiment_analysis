import sys
import pandas as pd
import numpy as np
import gensim.downloader as api
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'processed_data.pkl')
        self.model_path = os.path.join('artifacts', 'model.pkl')
        
        try:
            # Check if files exist before loading
            if not os.path.exists(self.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            # Load preprocessor and model
            with open(self.preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            self.model = load_object(file_path=self.model_path)
            
            # Load Word2Vec model from Gensim API
            logging.info("Loading Word2Vec model from Gensim API...")
            self.word2vec_model = api.load("word2vec-google-news-300")
            logging.info("Word2Vec model loaded successfully.")
        
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error("Error occurred while loading preprocessor, model, or Word2Vec model.")
            raise CustomException(e, sys)

    def predict(self, data):
        try:
            # Handle both CustomData and raw strings
            if isinstance(data, CustomData):
                df = data.get_data_as_dataframe()
            elif isinstance(data, str):
                df = CustomData(comment=data).get_data_as_dataframe()
            else:
                raise ValueError("Data must be of type 'CustomData' or 'str'.")

            # Preprocess the DataFrame
            comment_preprocessed = self.preprocessor.transform(df['reviewText'])
            logging.info("data transformation done")
        
            
            # Convert preprocessed comments into Word2Vec vectors
            vectors = []
            for comment in comment_preprocessed:
                tokens = comment.split()
                comment_vectors = [self.word2vec_model[token] for token in tokens if token in self.word2vec_model]
                
                if comment_vectors:
                    vectors.append(np.mean(comment_vectors, axis=0))
                else:
                    logging.warning("No vectors found for comment: {}".format(comment))
            
            if not vectors:
                raise ValueError("No valid vectors found for the given comments.")
            
            # Convert list of vectors into a NumPy array
            comment_vectors_array = np.array(vectors, dtype='float32')
            
            # Predict using the loaded model
            predictions = self.model.predict(comment_vectors_array)
            return predictions
        
        except ValueError as e:
            logging.error(f"Value error occurred: {e}")
            raise CustomException(e, sys)
        except Exception as e:
            logging.error('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self, comment: str):
        self.comment = comment

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'reviewText': [self.comment]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame created successfully')
            return df
        except Exception as e:
            logging.error('Exception occurred while creating DataFrame')
            raise CustomException(e, sys)
