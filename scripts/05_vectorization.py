'''
utf-8
05_vectorization.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- xx

This script vectorizes classification and object detection results to create vector representations per video. 
These representations are used for further classification and recommendation systems.

The script consists of:
    1. Importing necessary libraries.
    2. Loading configuration settings from a file.
    3. Creating vector representations for each video.
    4. Saving the results as a CSV file.

    --> Individual execution of the script: project_root/python -m scripts.05_vectorization
'''

# Imports
import os
import configparser
import pandas as pd
from collections import defaultdict
import numpy as np
from logger.logger_setup import setup_logger


# Logging
logger = setup_logger(name="Vectorization", log_file="logs/vectorization.log")


# config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
read_files = config.read(config_path)

detection_results_semantic_path = config['data']['detection_results_semantic_path']
output_vector_path = config['data']['output_vector_path']


# functions
def identify_classes(detection_path):
    '''
    Identify unique object classes from detection results and map them to indices.

    Args:
        - detection_path (str): Path to the detection results directory.

    Returns:
        - list: Sorted list of unique classes.
        - dict: Mapping of class names to indices.
    '''
    categories = [dr for dr in os.listdir(detection_path) if os.path.isdir(os.path.join(detection_path, dr))]
    all_classes = set()

    for category in categories:
        category_path = os.path.join(detection_path, category)
        for csv_file in os.listdir(category_path):
            if csv_file.endswith('.csv'):
                df = pd.read_csv(os.path.join(category_path, csv_file))
                if 'object' in df.columns:
                    df['object'] = df['object'].astype(str).fillna('unknown')
                    all_classes.update(df['object'].unique())
                else:
                    logger.warning(f"{csv_file} does not contain 'object' column.")

    sorted_classes = sorted(all_classes)
    class_to_idx = {c: i for i, c in enumerate(sorted_classes)}

    logger.info(f"Identified {len(sorted_classes)} unique classes.")
    return sorted_classes, class_to_idx


def calculate_video_vectors(detection_path, class_to_idx):
    '''
    Generate vector representations for videos based on object detection results.

    Args:
        - detection_path (str): Path to the detection results directory.
        - class_to_idx (dict): Mapping of class names to indices.

    Returns:
        - list: List of feature vectors for all videos.
        - list: List of corresponding video labels.
    '''
    video_features = []
    categories = [dr for dr in os.listdir(detection_path) if os.path.isdir(os.path.join(detection_path, dr))]

    for category in categories:
        category_path = os.path.join(detection_path, category)
        for csv_file in os.listdir(category_path):
            if csv_file.endswith('.csv'):
                video_name = csv_file.replace('.csv', '')
                df = pd.read_csv(os.path.join(category_path, csv_file))

                if 'object' in df.columns:
                    df['object'] = df['object'].astype(str).fillna('unknown')
                else:
                    logger.warning(f"The file {csv_file} does not contain the 'object' column. Defaulting to 'unknown'.")
                    df['object'] = 'unknown'

                vec = np.zeros(len(class_to_idx), dtype=float)

                class_groups = df.groupby('object')
                for obj, group in class_groups:
                    idx = class_to_idx.get(obj, None)
                    if idx is not None:
                        freq = len(group)
                        mean_conf = group['score'].mean() if 'score' in group.columns else 0.0
                        mean_area = group['area'].mean() if 'area' in group.columns else 0.0
                        importance = freq * mean_conf * (mean_area / (224 * 224))
                        vec[idx] = importance
                    else:
                        logger.warning(f"'{obj}' not found in class_to_idx.")

                video_features.append([category, video_name] + vec.tolist())

    logger.info(f"Calculated vectors for {len(video_features)} videos.")
    return video_features


# main
def main():
    '''
    Main function to process video features and save them to a CSV file.
    
    This function performs the following steps:
        1. Identifies classes from the detection results.
        2. Calculates video feature vectors based on the identified classes.
        3. Creates a DataFrame with the calculated video features.
        4. Saves the DataFrame to a CSV file.
    '''
    sorted_classes, class_to_idx = identify_classes(detection_results_semantic_path)

    video_features = calculate_video_vectors(detection_results_semantic_path, class_to_idx)

    columns = ['category', 'video_name'] + sorted_classes
    vectors_df = pd.DataFrame(video_features, columns=columns)

    vectors_df.to_csv(output_vector_path, index=False)
    logger.info(f"Saved feature vectors to {output_vector_path}")



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
