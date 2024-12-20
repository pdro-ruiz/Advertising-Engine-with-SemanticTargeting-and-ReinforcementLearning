'''
utf-8'
01_video_preprocessig.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- xx

This module is responsible for processing the videos and extracting frames at a resolution of 224x224 pixels with an interval of 3 fps.
The script goes through the directory structure specified in the configuration and extracts the frames from the videos into folders with the name of the parent folder and sequential numbering.

The script consists of:
1. Import of the necessary libraries.
2. Loading the configuration from a configuration file.
3. Definition of the extract_frames function that receives the video path, the destination directory, the fps and the image resolution.
4. Traversing the directory structure and extracting the frames from the videos.


    --> Individual execution of the script: project_root/python -m scripts.01_video_preprocessing
'''

# Imports
import sys
import os
import cv2
import configparser
from logger.logger_setup import setup_logger


# Logging
logger = setup_logger(name="VideoProcessing", log_file="logs/extract_frames.log")


# config
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
config.read(config_path)
if not config.sections():
    logger.error(f"No se pudo leer la configuración desde {config_path}. Secciones leídas: {config.sections()}")

raw_data_path = config['data']['raw_data_path']
processed_data_path = config['data']['processed_data_path']
frame_rate = int(config['preprocessing']['frame_rate'])
image_resolution = tuple(map(int, config['preprocessing']['image_resolution'].split(',')))

logger.info(f"Raw data path: {raw_data_path}")
logger.info(f"Processed data path: {processed_data_path}")
logger.info(f"Frame rate: {frame_rate}")
logger.info(f"Image resolution: {image_resolution}")


# functions
def extract_frames(video_path, target_dir, fps=frame_rate):
    '''
    Extract frames from a video file and save them to the specified target directory.

    Parameters:
        - video_path (str): Path to the input video file.
        - target_dir (str): Directory where the extracted frames will be saved.
        - fps (int): Frames per second rate at which frames will be extracted. Defaults to the video's original frame rate.
    '''
    logger.info(f"Processing video: {video_path}")            
    if not os.path.exists(target_dir):                                                 
        os.makedirs(target_dir, exist_ok=True)                                           
        logger.info(f"Directory created: {target_dir}")

    vidcap = cv2.VideoCapture(video_path)                                                
    if not vidcap.isOpened():                                                            
        logger.error(f"Error opening video: {video_path}")                                                 
        return

    frame_rate = int(vidcap.get(cv2.CAP_PROP_FPS))                                       
    frame_interval = frame_rate // fps                                                  

    success, image = vidcap.read()                                                       
    count = 0                                                                            
    while success:                                                                       
        if count % frame_interval == 0:                                                  
            h, w = image.shape[:2]                                                       
            scale = min(image_resolution[0]/w, image_resolution[1]/h)                    
            new_size = (int(w*scale), int(h*scale))                                      
            image_resized = cv2.resize(image, image_resolution)                          
            frame_file = os.path.join(target_dir, f"frame_{str(count).zfill(6)}.jpg")   
            cv2.imwrite(frame_file, image_resized)                                       
            logger.info(f"Frame saved: {frame_file}")
        success, image = vidcap.read()                                                  
        count += 1                                                                       

    vidcap.release()                                                                     
    logger.info(f"Processed {count} frames from video {video_path}")


def process_videos():
    '''
    Processes videos by iterating over all video categories and videos within them.
    '''
    logger.info(f"Exploring directory: {raw_data_path}")  
    for category in os.listdir(raw_data_path):                                               
        category_path = os.path.join(raw_data_path, category)                                
        if os.path.isdir(category_path):                                                     
            logger.info(f"Processing category: {category}")
            video_count = 0                                                                 
            for video_file in os.listdir(category_path):                                     
                if video_file.endswith('.mp4'):                                             
                    video_path = os.path.join(category_path, video_file)                    
                    video_count += 1                                                        
                    target_dir = os.path.join(processed_data_path, category, f"{category}_{str(video_count).zfill(6)}")  
                    logger.info(f"Processing video {video_count} from category {category}: {video_path}")  
                    extract_frames(video_path, target_dir)                                   

    logger.info(f"Video processing completed.")


if __name__ == "__main__":
    try:
        process_videos()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)