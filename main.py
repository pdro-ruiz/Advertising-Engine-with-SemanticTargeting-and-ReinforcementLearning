'''
utf-8
main.py

-------------------------------------------------------------------------------------------------- 
Author: Pedro Ruiz
Creation: 12-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- 

This script orchestrates the execution of the multiple steps of the deep learning pipeline.
It reads the configuration file, configures the registry and sequentially executes each script.

The steps include:
    1. Video preprocessing
    2. Transfer learning
    3. Semantic segmentation
    4. Vectorization
    5. Video classification
    6. Video recommendation
    7. Reinforcement learning-based ad recommendation
'''

# imports
import subprocess
import configparser
import os
import sys
from logger.logger_setup import setup_logger

# config
config = configparser.ConfigParser()
config.read('config/settings.conf')

# logging
log_file = config['logging'].get('log_file', 'logs/pipeline.log')
logger = setup_logger(name="PipelineOrchestration", log_file=log_file)


def run_step(script_name):
    '''
    This function runs the specified script as a subprocess.
    '''
    logger.info(f"Running the script: {script_name}")
    module_name = script_name.replace('.py', '')
    result = subprocess.run([sys.executable, '-m', f"scripts.{module_name}"], text=True)
    
    if result.returncode != 0:
        logger.error(f"Error executing {script_name}: {result.stderr}")
        raise RuntimeError(f"Failure in {script_name}")
    else:
        logger.info(f"{script_name} successfully completed.")

# main
def main():
    steps = [
        "01_video_preprocessing.py",
        "02_transfer_learning.py",
        "04_Semantic_segmentation.py",
        "05_vectorization.py",
        "06_video_classifier.py",
        "07_video_recommender.py",
        "08_rl_ad_recommender.py"
    ]
    logger.info("Starting pipeline...")
    try:
        for step in steps:
            run_step(step)
        logger.info("Pipeline successfully completed.")
    except Exception as e:
        logger.error(f"Pipeline stopped: {e}")

if __name__ == "__main__":
    main()