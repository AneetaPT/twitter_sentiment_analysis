import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
import pickle
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import gensim
import gensim.downloader as api

nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.base import BaseEstimator, TransformerMixin

@dataclass
class DataPreprocessorConfig:
    preprocessor_pipeline_file_path: str = os.path.join('artifacts', 'processed_data.pkl')

class PreprocessTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df = df.str.lower()
        df = df.apply(lambda x: re.sub('[^a-z A-Z 0-9-]+', '', str(x)))
        stop_words = set(stopwords.words('english'))
        df = df.apply(lambda x: " ".join([y for y in x.split() if y not in stop_words]))
        df = df.apply(lambda x: re.sub(r'(http|https|ftp|ssh)://[\S]+', '', str(x)))
        df = df.apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
        df = df.apply(lambda x: " ".join(x.split()))
        return df

class LemmatizeTextTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        lm = WordNetLemmatizer()
        df = X.copy()
        df = df.apply(lambda x: " ".join([lm.lemmatize(word, pos='v') for word in x.split()]))
        return df

class DataPreprocessor:
    def __init__(self):
        self.config = DataPreprocessorConfig()

    def clean_data(self, df: pd.DataFrame):
        try:
            logging.info("Starting data cleaning...")
            df = df.drop(['2401', 'Borderlands'], axis=1)
            df.columns = ['review', 'reviewText']
            df.dropna(inplace=True)
            df = df[df['reviewText'].apply(len) > 1]
            df = df[df['review'] != 'Irrelevant']
            df = df[df['review'] != 'Neutral']
            df = pd.get_dummies(df, columns=['review'])
            df = df.drop(['review_Positive'], axis=1)
            df['review_Negative'] = df['review_Negative'].astype(int)
            logging.info("Data cleaning completed.")
            return df
        except Exception as e:
            logging.exception("Error occurred during data cleaning.")
            raise CustomException(e, sys)
    
    def split_data(self, df: pd.DataFrame):
        try:
            logging.info("Splitting data into train and test sets...")
            X = df['reviewText']
            y = df.drop(['reviewText'], axis=1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            logging.info("Data splitting completed.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.exception("Error occurred during data splitting.")
            raise CustomException(e, sys)

    def create_and_save_pipeline(self):
        try:
            logging.info("Creating preprocessing pipeline...")
            pipeline = Pipeline([
                ('preprocess_text', PreprocessTextTransformer()),
                ('lemmatize_text', LemmatizeTextTransformer())
            ])

            # Save the pipeline as a .pkl file
            save_object(self.config.preprocessor_pipeline_file_path, pipeline)
            logging.info("Preprocessing pipeline created and saved successfully.")
        except Exception as e:
            logging.exception("Error occurred while creating and saving preprocessing pipeline.")
            raise CustomException(e, sys)


@dataclass
class Word2VecProcessorConfig:
    model_name: str = "word2vec-google-news-300"


class Word2VecProcessor:
    def __init__(self):
        try:
            logging.info("Loading Word2Vec model...")
            self.wv = api.load("word2vec-google-news-300")
            self.config = Word2VecProcessorConfig()
            logging.info("Word2Vec model loaded successfully.")
        except Exception as e:
            logging.exception("Error occurred while loading Word2Vec model.")
            raise CustomException(e, sys)

    def preprocess(self, sentences):
        try:
            logging.info("Starting Word2Vec preprocessing...")
            vectors = []
            valid_indices = []
            for i, sentence in enumerate(sentences):
                tokens = sentence.split()
                sentence_vector = [self.wv[token] for token in tokens if token in self.wv]
                if sentence_vector:
                    vectors.append(np.mean(sentence_vector, axis=0))
                    valid_indices.append(i)
            logging.info("Word2Vec preprocessing completed.")
            return np.array(vectors, dtype='float32'), valid_indices
        except Exception as e:
            logging.exception("Error occurred during Word2Vec preprocessing.")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, df):
        df = pd.read_csv(df)

        # Step 1: Data Preprocessing
        preprocessor = DataPreprocessor()

        try:
            df = preprocessor.clean_data(df)
            X_train, X_test, y_train, y_test = preprocessor.split_data(df)
            preprocessor.create_and_save_pipeline()

            # Load and use the pipeline
            pipeline = pickle.load(open(preprocessor.config.preprocessor_pipeline_file_path, 'rb'))
            X_train = pipeline.transform(X_train)
            X_test = pipeline.transform(X_test)
            
            # Save the pipeline
        except Exception as e:
            logging.exception("Error occurred in data preprocessing pipeline.")
            raise CustomException(e, sys)

        # Step 2: Vectorization using Word2Vec
        w2v_processor = Word2VecProcessor()

        try:
            X_train_vectors, train_valid_indices = w2v_processor.preprocess(X_train.tolist())
            X_test_vectors, test_valid_indices = w2v_processor.preprocess(X_test.tolist())

            # Adjust y_train and y_test to only include valid indices
            Y_train = [y_train.iloc[i]['review_Negative'] for i in train_valid_indices]
            Y_test = [y_test.iloc[i]['review_Negative'] for i in test_valid_indices]

            return X_train,X_test,y_train,y_test,X_train_vectors, X_test_vectors, Y_train, Y_test
        except Exception as e:
            logging.exception("Error occurred in Word2Vec processing pipeline.")
            raise CustomException(e, sys)