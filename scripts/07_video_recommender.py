'''utf-8'
07_video_recommender.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- xx

This script develops a video recommendation system to identify similar videos based on their feature representations.

The script consists of:
    1. Loading the dataset with video features.
    2. Defining a function to recommend similar videos based on cosine similarity.
    3. Recommending similar videos for an example video.
    
    --> Individual execution of the script: project_root/python -m scripts.07_video_recommender

'''

# imports
import os
import configparser
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from logger.logger_setup import setup_logger


# Logging
logger = setup_logger(name="VideoRecommender", log_file="logs/video_recommender.log")


# config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
config.read(config_path)

features_path = config['data']['output_vector_path']


# functions
def load_data(file_path):
    '''
    This function loads the dataset from the specified file and returns the data, features, video names, and video categories.
    '''
    logger.info(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)
    X = df.drop(columns=['category', 'video_name']).values
    video_names = df['video_name'].values
    video_categories = df['category'].values
    logger.info(f"Loaded dataset with {X.shape[0]} videos and {X.shape[1]} features.")
    return df, X, video_names, video_categories


def recommend_similar(df, X, video_names, video_name, top_k=10):
    '''
    Recommends similar videos to the specified video based on cosine similarity.
    
    Parameters:
        - df (pd.DataFrame): DataFrame containing video data.
        - X (np.ndarray): Feature matrix where each row corresponds to a video.
        - video_names (np.ndarray): Array of video names corresponding to the rows in X.
        - video_name (str): The name of the video for which to find similar videos.
        - top_k (int, optional): The number of similar videos to return. Default is 10.
        
    Returns:
        - DataFrame containing the top_k similar videos with their categories and names.
    '''
    if video_name not in video_names:
        logger.error(f"Video '{video_name}' not found in the dataset.")
        return pd.DataFrame()

    idx = np.where(video_names == video_name)[0][0]
    query = X[idx].reshape(1, -1)
    dists = cosine_distances(query, X)[0]
    nearest = np.argsort(dists)
    nearest = nearest[nearest != idx]
    top_indices = nearest[:top_k]
    results = df.iloc[top_indices][['category', 'video_name']]
    logger.info(f"Found {len(results)} similar videos for '{video_name}'.")
    return results


# main
def main():
    '''
    Main function to load video data, log the input video and its category, and recommend similar videos.
    Steps:
        1. Load data including video features, names, and categories.
        2. Log the input video and its category.
        3. Recommend and log similar videos.
    '''
    
    df, X, video_names, video_categories = load_data(features_path)
    example_video = video_names[0]
    logger.info(f"Input video: {example_video}, Category: {video_categories[0]}")
    similar_videos = recommend_similar(df, X, video_names, example_video, top_k=5)
    logger.info("Recommended videos:")
    logger.info(f"\n{similar_videos}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
