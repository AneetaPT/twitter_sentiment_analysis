import sys
import pandas as pd
import numpy as np
import gensim.downloader as api
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')
        self.tokenizer_path = os.path.join('artifacts', 'tokenizer.pkl')  # Path for the tokenizer
        
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")

            self.model = load_object(file_path=self.model_path)

            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)  # Load the tokenizer
           
        except Exception as e:
            logging.error("Error occurred while loading preprocessor, model, or Word2Vec model.")
            raise CustomException(e, sys)

    def predict(self, data):
        try:
            # Check if the input is an instance of CustomData
            if isinstance(data, CustomData):
                comment = data.comment  # Get the comment attribute
            elif isinstance(data, str):
                comment = data  # If it's a string, use it directly
            else:
                raise ValueError("Data must be of type 'CustomData' or 'str'.")

            # Convert the received comment into a list (required for tokenization)
            input_text = [comment]   # Wrap the string comment in a list

            max_sequence_length = 50  # Ensure this matches your model's expected input length

            # Tokenize and pad the input text
            input_seq = self.tokenizer.texts_to_sequences(input_text)  # Use the loaded tokenizer
            input_padded = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')  # Padding

            # Predict using the loaded GRU model
            predictions = self.model.predict(input_padded)

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
