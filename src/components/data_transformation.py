import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

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


@dataclass
class DataPreprocessorConfig:
    processed_data_file_path: str = os.path.join('artifacts', 'processed_data.pkl')
    X_train_file_path: str = os.path.join('artifacts', 'X_train.csv')
    X_test_file_path: str = os.path.join('artifacts', 'X_test.csv')
    y_train_file_path: str = os.path.join('artifacts', 'y_train.csv')
    y_test_file_path: str = os.path.join('artifacts', 'y_test.csv')


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

    def preprocess_text(self, df: pd.DataFrame):
        try:
            logging.info("Starting text preprocessing...")
            df['reviewText'] = df['reviewText'].str.lower()
            df['reviewText'] = df['reviewText'].apply(lambda x: re.sub('[^a-z A-Z 0-9-]+', '', str(x)))
            stop_words = set(stopwords.words('english'))
            df['reviewText'] = df['reviewText'].apply(
                lambda x: " ".join([y for y in x.split() if y not in stop_words]))
            df['reviewText'] = df['reviewText'].apply(
                lambda x: re.sub(r'(http|https|ftp|ssh)://[\S]+', '', str(x)))
            df['reviewText'] = df['reviewText'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
            df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x.split()))
            logging.info("Text preprocessing completed.")
            return df
        except Exception as e:
            logging.exception("Error occurred during text preprocessing.")
            raise CustomException(e, sys)

    def lemmatize_text(self, df: pd.DataFrame):
        try:
            logging.info("Starting text lemmatization...")
            lm = WordNetLemmatizer()
            df['reviewText'] = df['reviewText'].apply(
                lambda x: " ".join([lm.lemmatize(word, pos='v') for word in x.split()]))
            logging.info("Text lemmatization completed.")
            return df
        except Exception as e:
            logging.exception("Error occurred during text lemmatization.")
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

    def save_processed_data(self, df: pd.DataFrame, X_train, X_test, y_train, y_test):
        try:
            logging.info("Saving processed data and train/test splits...")
            save_object(

                file_path=self.config.processed_data_file_path,
                obj=df

            )
            X_train.to_csv(self.config.X_train_file_path, index=False)
            X_test.to_csv(self.config.X_test_file_path,index=False)
            y_train.to_csv(self.config.y_train_file_path,index=False)
            y_test.to_csv(self.config.y_test_file_path,index=False)
            logging.info("Processed data and splits saved successfully.")
            return (
                X_train,
                X_test,
                y_train,
                y_test,
                self.config.processed_data_file_path
            )
        except Exception as e:
            logging.exception("Error occurred while saving processed data.")
            raise CustomException(e, sys)


@dataclass
class Word2VecProcessorConfig:
    model_name: str = "word2vec-google-news-300"
    X_train_vectors_file_path: str = os.path.join('artifacts', 'X_train_vectors.npy')
    X_test_vectors_file_path: str = os.path.join('artifacts', 'X_test_vectors.npy')
    y_train_valid_file_path: str = os.path.join('artifacts', 'y_train_valid.csv')
    y_test_valid_file_path: str = os.path.join('artifacts', 'y_test_valid.csv')


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

    def save_vectors_and_labels(self, X_train_vectors, X_test_vectors, Y_train, Y_test):
        try:
            logging.info("Saving Word2Vec vectors and labels...")
            save_object(self.config.X_train_vectors_file_path, X_train_vectors)
            save_object(self.config.X_test_vectors_file_path, X_test_vectors)
            pd.DataFrame(Y_train).to_csv(self.config.y_train_valid_file_path, index=False, header=False)
            pd.DataFrame(Y_test).to_csv(self.config.y_test_valid_file_path, index=False, header=False)
            logging.info("Word2Vec vectors and labels saved successfully.")
            return (
                X_train_vectors,
                X_test_vectors,
                Y_train,
                Y_test,
                self.config.X_train_vectors_file_path,
                self.config.X_test_vectors_file_path,
                self.config.y_train_valid_file_path,
                self.config.y_test_valid_file_path
            )
        except Exception as e:
            logging.exception("Error occurred while saving Word2Vec vectors and labels.")
            raise CustomException(e, sys)


    def initiate_data_transformation(self,df):

        df = pd.read_csv(df) 

        # Step 1: Data Preprocessing
        preprocessor = DataPreprocessor()

        try:
            df = preprocessor.clean_data(df)
            df = preprocessor.preprocess_text(df)
            df = preprocessor.lemmatize_text(df)
            X_train, X_test, y_train, y_test = preprocessor.split_data(df)
            train_data, test_data, train_labels, test_labels, processed_data_path =preprocessor.save_processed_data(
                df, X_train, X_test, y_train, y_test
            )
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

            # Save vectors and labels
            train_vectors, test_vectors, Y_train, Y_test, train_vectors_path, test_vectors_path, train_labels_path, test_labels_path = w2v_processor.save_vectors_and_labels(
                X_train_vectors, X_test_vectors, Y_train, Y_test
            )
            return (train_vectors,
                    test_vectors,
                    Y_train, 
                    Y_test
            )
        except Exception as e:
            logging.exception("Error occurred in Word2Vec processing pipeline.")
        