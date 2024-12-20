'''utf-8'
03_feature_extraction_classification.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- xx

This script extracts features and classifies video frames using a fine-tuned ResNet50 model. 
It replicates the directory hierarchy of `processed/` in `detection_results/`, ensuring a clear and organized structure for results.

The script consists of:
    1. Importing necessary libraries.
    2. Loading configuration settings from a file.
    3. Loading the fine-tuned ResNet50 model.
    4. Preprocessing images.
    5. Classifying frames and saving results to CSV files.

    --> Individual execution of the script: project_root/python -m scripts.03_feature_extraction_classification
'''

# Imports
import os
import configparser
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.models as models
from logger.logger_setup import setup_logger


# Logging
logger = setup_logger(name="FeatureExtraction", log_file="logs/feature_extraction.log")

# config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
config.read(config_path)

processed_data_path = config['data']['processed_data_path']
detection_results_path = config['data']['detection_results_path']
os.makedirs(detection_results_path, exist_ok=True)

transfers_path = config['data']['transfers_path']
best_model_path = os.path.join(transfers_path, 'resnet_finetuned.pth')

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

checkpoint = torch.load(best_model_path, map_location=device)
class_names = checkpoint['class_names']

model = models.resnet50(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval().to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# function
def process_video(video_dir, output_csv_path):
    '''
    Processes video frames stored in a directory, performs classification on each frame, and saves the results to a CSV file.
    
    Args:
        - video_dir (str): Path to the directory containing video frames as .jpg files.
        - output_csv_path (str): Path to save the output CSV file containing classification results.
    '''
    
    frames = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
    frames.sort()

    results = []
    for frame_name in tqdm(frames, desc=f"Processing {os.path.basename(video_dir)}"):
        frame_path = os.path.join(video_dir, frame_name)
        img = Image.open(frame_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_t)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            predicted_class = class_names[pred.item()]
            results.append([frame_name, predicted_class, conf.item()])

    df = pd.DataFrame(results, columns=["frame_id", "predicted_class", "confidence"])
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved CSV to {output_csv_path}")

def main():
    '''
    Main function to explore processed data directories, process each category and video, and perform feature extraction and classification.
        1. Logs the start of exploration of the processed data path.
        2. Iterates through each category in the processed data path.
        3. For each category, logs the processing status and creates an output directory.
        4. Iterates through each video directory within the category.
        5. For each video directory, processes the video and saves the results to a CSV file.
        6. Logs the completion of feature extraction and classification.
    '''
    
    logger.info(f"Exploring {processed_data_path}")
    for category in os.listdir(processed_data_path):
        category_path = os.path.join(processed_data_path, category)
        if os.path.isdir(category_path):
            logger.info(f"Processing category: {category}")

            category_output_dir = os.path.join(detection_results_path, category)
            os.makedirs(category_output_dir, exist_ok=True)

            for video_dir in os.listdir(category_path):
                video_path = os.path.join(category_path, video_dir)
                if os.path.isdir(video_path):
                    output_csv = os.path.join(category_output_dir, f"{video_dir}.csv")
                    process_video(video_path, output_csv)

    logger.info("Feature extraction and classification completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
