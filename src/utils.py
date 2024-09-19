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


def tokenize_and_pad(X_train, X_test, num_words=5000, maxlen=50):
    try:
        if isinstance(X_train, np.ndarray):
            X_train = X_train.astype(str).tolist()
        if isinstance(X_test, np.ndarray):
            X_test = X_test.astype(str).tolist()
        tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_test_seq = tokenizer.texts_to_sequences(X_test)

        X_train_padded = pad_sequences(X_train_seq, maxlen=maxlen, padding='post', truncating='post')
        X_test_padded = pad_sequences(X_test_seq, maxlen=maxlen, padding='post', truncating='post')

        return X_train_padded, X_test_padded
    except Exception as e:
        raise CustomException(e, sys)
