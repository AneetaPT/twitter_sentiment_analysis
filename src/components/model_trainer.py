import os
import sys
from dataclasses import dataclass
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout

from src.exception import CustomException
from src.logger import logging
from src.utils import tokenize_and_pad, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,X_train,X_test,y_train,y_test,X_train_vectors, X_test_vectors, Y_train, Y_test):
        
        try:
            # Tokenize and pad for deep learning models
            X_train_padded, X_test_padded = tokenize_and_pad(X_train, X_test)


           # Train and evaluate traditional models
            logging.info("Training Gaussian Naive Bayes model")
            gnb_model = GaussianNB().fit(X_train_vectors, Y_train)
            gnb_accuracy = accuracy_score(Y_test, gnb_model.predict(X_test_vectors))
            logging.info(f"GaussianNB Test Accuracy: {gnb_accuracy}")

            logging.info("Training Random Forest Classifier model")
            rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_classifier.fit(X_train_vectors, Y_train)
            rf_accuracy = accuracy_score(Y_test, rf_classifier.predict(X_test_vectors))
            logging.info(f"Random Forest Test Accuracy: {rf_accuracy}")

            # Build, train, and evaluate deep learning models
            try:
                logging.info("Building and training LSTM model")
                lstm_model = self.build_lstm_model()
                history1=lstm_model.fit(X_train_padded, y_train, epochs=10, batch_size=32)
                loss, lstm_accuracy =lstm_model.evaluate(X_test_padded, y_test)
                logging.info(f"LSTM Test Accuracy: {lstm_accuracy}")
            except Exception as e:
                logging.error(f"Error during LSTM model training: {e}")
                raise  # Re-raise the exception after logging

            logging.info("Building and training GRU model")
            gru_model = self.build_gru_model()
            history2=gru_model.fit(X_train_padded, y_train, epochs=15, batch_size=32, validation_split=0.1)
            loss, gru_accuracy =gru_model.evaluate(X_test_padded, y_test)
            logging.info(f"GRU Test Accuracy: {gru_accuracy}")

            # Compare accuracies and save the best model
            model_accuracies = {
                "GaussianNB": (gnb_model, gnb_accuracy),
                "RandomForest": (rf_classifier, rf_accuracy),
                "LSTM": (lstm_model, lstm_accuracy),
                "GRU": (gru_model, gru_accuracy),
            }

            best_model_name, (best_model, best_accuracy) = max(model_accuracies.items(), key=lambda item: item[1][1])

            logging.info(f"Best Model: {best_model_name} with Accuracy: {best_accuracy}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            print(best_model_name,best_accuracy)
            return best_model_name, best_accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def build_lstm_model(self):
        vocab_size = 5000
        embedding_dim = 64
        max_length = 50

        model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            LSTM(64, return_sequences=True),
            Dropout(0.5),
            LSTM(32),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_gru_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=50))
        model.add(GRU(128, return_sequences=True))
        model.add(GRU(128))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
