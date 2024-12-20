'''
utf-8
06_video_classifier.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- xx

This script implements a classifier to categorize videos based on their extracted features.

This script consists of:
    1. Import of the necessary libraries.
    2. Loading the dataset.
    3. Training the Logistic Regression model.
    4. Evaluating the model.
    
    --> Individual execution of the script: project_root/python -m scripts.06_video_classifier
'''

# imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from logger.logger_setup import setup_logger
import os
import configparser


# logging
logger = setup_logger(name="VideoClassifier", log_file="logs/video_classifier.log")


# config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
config.read(config_path)

dataset_path = config['data']['output_vector_path']


# functions
def load_data(file_path):
    '''
    Load dataset from a CSV file.
    This function reads a CSV file containing video data, extracts the features and labels,
    and returns them as separate arrays.
    '''
    logger.info(f"Loading dataset from {file_path}")
    vectors_df = pd.read_csv(file_path)
    X = vectors_df.drop(columns=['category', 'video_name']).values
    y = vectors_df['category'].values
    logger.info(f"Loaded dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    return X, y


def train_model(X_train, y_train):
    '''
    This function trains a Logistic Regression model using the provided training data.
    '''
    logger.info("Training Logistic Regression model.")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    logger.info("Model training completed.")
    return clf


def evaluate_model(clf, X_test, y_test):
    '''
    This function evaluates the trained model using the provided test data.
    '''
    logger.info("Evaluating model.")
    y_pred = clf.predict(X_test)
    logger.info("Classification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))
    conf_matrix = confusion_matrix(y_test, y_pred)
    logger.info("Confusion Matrix:")
    logger.info(f"\n{conf_matrix}")


# main
def main():
    '''
    Main function to load data, split it into training and testing sets,
    train a model, and evaluate its performance.
    '''
    X, y = load_data(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = train_model(X_train, y_train)
    evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
