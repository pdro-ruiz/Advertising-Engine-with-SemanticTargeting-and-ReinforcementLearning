'''
utf-8
08_rl_ad_recommender.py

--------------------------------------------------------------------------------------------------
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
--------------------------------------------------------------------------------------------------

This script implements an advertisement recommendation system using Reinforcement Learning (Q-learning).

The script consists of:
    1. Loading the dataset with video features.
    2. Defining functions to recommend similar videos and simulate an environment step.
    3. Training a Q-learning model to recommend the best ad for each video category.
    4. Logging the best ad for each category.
    
    --> Individual execution of the script: project_root/python -m scripts.08_rl_ad_recommender
'''

# imports
import os
import configparser
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import random
from logger.logger_setup import setup_logger


# Logging
logger = setup_logger(name="AdRecommender", log_file="logs/ad_recommender.log")


# config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
config.read(config_path)

features_file = config['data']['output_vector_path']
num_episodes = int(config['q_learning']['num_episodes'])
alpha = float(config['q_learning']['alpha'])
gamma = float(config['q_learning']['gamma'])
epsilon = float(config['q_learning']['epsilon'])

# functions
def load_data(config_path, features_file):
    '''
    This function loads the dataset from the specified file and returns the data, features, video names, and video categories.
    '''
    df = pd.read_csv(features_file)
    X = df.drop(columns=['category', 'video_name']).values
    video_names = df['video_name'].values
    video_categories = df['category'].values
    logger.info(f"Loaded dataset with {len(video_names)} videos and {X.shape[1]} features.")
    return df, X, video_names, video_categories


def recommend_similar(df, video_names, X, video_name, top_k=10):
    '''
    Recommend similar videos based on cosine similarity.

    Parameters:
        - df (pd.DataFrame): DataFrame containing video information.
        - video_names (np.ndarray): Array of video names corresponding to the rows in X.
        - X (np.ndarray): Feature matrix where each row corresponds to a video.
        - video_name (str): The name of the video for which to find similar videos.
        - top_k (int, optional): The number of similar videos to return. Default is 10.

    Returns:
        - pd.DataFrame: DataFrame containing the top_k similar videos with their categories and names.
    '''

    idx = np.where(video_names == video_name)[0]
    if len(idx) == 0:
        logger.error(f"Video '{video_name}' not found.")
        return pd.DataFrame()
    idx = idx[0]
    query = X[idx].reshape(1, -1)
    dists = cosine_distances(query, X)[0]
    nearest = np.argsort(dists)
    nearest = nearest[nearest != idx]
    top_indices = nearest[:top_k]
    return df.iloc[top_indices][['category', 'video_name']]


def env_step(df, categories, ad_info, current_category, action, video_names, X, top_k=5):
    '''
    Simulates an environment step by selecting an action and returning the next state and reward.
    
    Parameters:
        - df (pd.DataFrame): DataFrame containing video data with at least 'category' and 'video_name' columns.
        - categories (list): List of advertisement categories.
        - ad_info (dict): Dictionary containing advertisement information with probabilities and rewards.
        - current_category (str): The current category of the video being watched.
        - action (int): The action taken by the agent, which determines the advertisement category and type.
        - video_names (list): List of video names.
        - X (np.ndarray): Feature matrix for the videos.
        - top_k (int, optional): Number of top similar videos to consider for recommendation. Default is 5.
        
    Returns:
        - tuple: A tuple containing the next category (str) and the reward (float).
    '''
    cat_idx = action // 2
    tipo_idx = action % 2
    ad_cat = categories[cat_idx]
    ad_type = 'short' if tipo_idx == 0 else 'long'

    p, r = ad_info[ad_cat][ad_type]
    user_view = np.random.rand()
    reward = r if user_view < p else 0.0

    vids_cat = df[df['category'] == current_category]['video_name'].values
    current_video = np.random.choice(vids_cat)
    sim_vids = recommend_similar(df, video_names, X, current_video, top_k=top_k)

    if len(sim_vids) == 0:
        next_category = current_category
    else:
        chosen = sim_vids.sample(1).iloc[0]
        next_category = chosen['category']

    return next_category, reward


def train_q_learning(df, categories, ad_info, video_names, X, num_episodes, alpha, gamma, epsilon):
    '''
    Trains a Q-learning model to recommend the best ad for each video category.
    
    Parameters:
        - df (pd.DataFrame): DataFrame containing the data.
        - categories (list): List of video categories.
        - ad_info (dict): Dictionary containing information about ads.
        - video_names (list): List of video names.
        - X (np.array): Feature matrix.
        - num_episodes (int, optional): Number of episodes for training. Default is 1000.
        - alpha (float, optional): Learning rate. Default is 0.1.
        - gamma (float, optional): Discount factor. Default is 0.9.
        - epsilon (float, optional): Exploration rate. Default is 0.1.
        
    Returns:
        - tuple: A tuple containing:
            + Q (np.array): The Q-table learned by the algorithm.
            + rewards_per_episode (list): List of total rewards per episode.
    '''
    num_states = len(categories)
    num_actions = num_states * 2
    Q = np.zeros((num_states, num_actions))
    category_to_idx = {c: i for i, c in enumerate(categories)}

    rewards_per_episode = []
    for episode in range(num_episodes):
        state_idx = np.random.randint(num_states)
        state_category = categories[state_idx]
        total_reward = 0

        for _ in range(10):
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state_idx])

            next_category, reward = env_step(df, categories, ad_info, state_category, action, video_names, X)
            next_state_idx = category_to_idx[next_category]

            Q[state_idx, action] += alpha * (reward + gamma * np.max(Q[next_state_idx]) - Q[state_idx, action])
            total_reward += reward
            state_idx = next_state_idx

        rewards_per_episode.append(total_reward)

    logger.info("Q-learning training completed.")
    return Q, rewards_per_episode


# main
def main():
    '''
    Main function to load data, train a Q-learning model, and log the best ad recommendations for each video category.
    
        1. Load data and extract features and video categories.
        2. Define ad information for different categories and types.
        3. Train a Q-learning model using the loaded data and ad information.
        4. Determine and log the best ad type for each video category based on the trained Q-learning model.
    '''
    
    df, X, video_names, video_categories = load_data(config_path, features_file)
    categories = np.unique(video_categories)

    ad_info = {
        'cooking': {'long': (0.3, 3), 'short': (0.7, 1)},
        'traffic': {'long': (0.4, 2), 'short': (0.8, 1)},
        'videoclip': {'long': (0.5, 2), 'short': (0.9, 1)}
    }

    Q, rewards_per_episode = train_q_learning(
        df, categories, ad_info, video_names, X,
        num_episodes=num_episodes, alpha=alpha, gamma=gamma, epsilon=epsilon
    )
    
    for i, cat in enumerate(categories):
        best_action = np.argmax(Q[i])
        cat_ad = best_action // 2
        ad_type = best_action % 2
        chosen_cat = categories[cat_ad]
        chosen_type = 'short' if ad_type == 0 else 'long'
        logger.info(f"For category {cat}, best ad: {chosen_type} from {chosen_cat}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
