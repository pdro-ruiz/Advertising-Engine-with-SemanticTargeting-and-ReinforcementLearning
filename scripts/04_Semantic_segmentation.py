'''utf-8'
04_Semantic_segmentation.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- xx

This script performs object detection using a pre-trained Mask R-CNN model on the COCO dataset. 
The results for each video are saved as CSV files, including detected objects, confidence scores, and bounding box areas.

The script consists of:
    1. Importing necessary libraries.
    2. Loading configuration settings from a file.
    3. Defining the process_video function to perform object detection on each frame of a video.
    4. Iterating over the processed data directory and performing object detection on each video.
    5. Saving the results as CSV files. 
    
    --> Individual execution of the script: project_root/python -m scripts.04_Semantic_segmentation

'''

# Imports
import os
import configparser
import torch
from PIL import Image
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision import transforms
import pandas as pd
from tqdm import tqdm
from logger.logger_setup import setup_logger


# Logging
logger = setup_logger(name="SemanticSegmentation", log_file="logs/semantic_segmentation.log")


# Config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
config.read(config_path)

processed_data_path = config['data']['processed_data_path']
detection_results_semantic_path = config['data']['detection_results_semantic_path']
os.makedirs(detection_results_semantic_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load pre-trained Mask R-CNN model
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights)
model.eval().to(device)

transform = transforms.ToTensor()

coco_labels = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


# Functions
def process_video(video_dir, output_csv_path):
    '''
    Process a video by performing object detection on each frame using a pre-trained Mask R-CNN model.
    This function iterates through each frame in the specified video directory, applies object detection,
    and saves the results to a CSV file. The CSV file contains the following columns: frame_id, object, score, area.
    
    Parameters:
        - video_dir (str): The directory containing the video frames as .jpg files.
        - output_csv_path (str): The path where the resulting CSV file will be saved.
    '''
    frames = [f for f in os.listdir(video_dir) if f.endswith('.jpg')]
    frames.sort()

    results = []
    for frame_name in tqdm(frames, desc=f"Processing {os.path.basename(video_dir)}"):
        frame_path = os.path.join(video_dir, frame_name)
        img = Image.open(frame_path).convert("RGB")
        img_t = transform(img).to(device).unsqueeze(0)

        with torch.no_grad():
            prediction = model(img_t)[0]

        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']

        score_threshold = 0.5
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            if score >= score_threshold:
                class_id = label.item()
                class_name = coco_labels[class_id] if class_id < len(coco_labels) else f"class_{class_id}"
                x1, y1, x2, y2 = box.tolist()
                w = x2 - x1
                h = y2 - y1
                area_box = w * h
                results.append([frame_name, class_name, score.item(), area_box])

    df = pd.DataFrame(results, columns=["frame_id", "object", "score", "area"])
    df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved CSV to {output_csv_path}")


# main
def main():
    '''
    Main function to perform object detection with semantic segmentation.
    This function explores the processed data directory, iterates through each category,
    and processes each video directory within the category. 
    The results are saved as CSV files in the specified output directory.
    
    Steps:
        1. Logs the start of the exploration of the processed data path.
        2. Iterates through each category in the processed data path.
        3. For each category, creates an output directory if it doesn't exist.
        4. Iterates through each video directory within the category.
        5. Processes each video and saves the results as a CSV file in the category's output directory.
        6. Logs the completion of the object detection with semantic segmentation.
    Note:
    
        - The function assumes that `processed_data_path` and `detection_results_semantic_path` are predefined.
        - The `process_video` function is called to handle the video processing.
    '''
    logger.info(f"Exploring {processed_data_path}")
    for category in os.listdir(processed_data_path):
        category_path = os.path.join(processed_data_path, category)
        if os.path.isdir(category_path):
            category_output_dir = os.path.join(detection_results_semantic_path, category)
            os.makedirs(category_output_dir, exist_ok=True)

            for video_dir in os.listdir(category_path):
                video_path = os.path.join(category_path, video_dir)
                if os.path.isdir(video_path):
                    output_csv = os.path.join(category_output_dir, f"{video_dir}.csv")
                    process_video(video_path, output_csv)

    logger.info("Object detection with semantic segmentation completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
