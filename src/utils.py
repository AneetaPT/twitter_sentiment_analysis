import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


from src.exception import CustomException

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def tokenize_and_pad(X_train, X_test, num_words=5000, maxlen=50, tokenizer_path='artifacts/tokenizer.pkl'):
    try:
        # Load tokenizer if it exists, otherwise create a new one
        tokenizer_path: str = os.path.join('artifacts', 'tokenizer.pkl')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as handle:
                tokenizer = pickle.load(handle)
        else:
            tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
            tokenizer.fit_on_texts(X_train)
            # Save the tokenizer for future use
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Tokenize the training and testing data
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        # Pad sequences to ensure uniform input size
        X_train_padded = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

        return X_train_padded, X_test_padded
    except Exception as e:
        raise CustomException(e, sys)