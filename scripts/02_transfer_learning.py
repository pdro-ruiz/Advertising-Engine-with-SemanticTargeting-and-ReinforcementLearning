'''utf-8'
02_transfer_learning.py

-------------------------------------------------------------------------------------------------- xx
Author: Pedro Ruiz
Creation: 09-12-2024
Version: 1.0
-------------------------------------------------------------------------------------------------- xx

This script fine-tunes a pre-trained ResNet50 model on a custom dataset with reduced data, demonstrating its effectiveness as a classifier.

The script consists of:
    1. Importing necessary libraries.
    2. Loading configuration settings from a file.
    3. Preprocessing the dataset.
    4. Setting up the ResNet50 model for fine-tuning.
    5. Training and validating the model.
    6. Saving the best-performing model.
    
    --> Individual execution of the script: project_root/python -m scripts.02_transfer_learning
'''

# Imports
import os
import configparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import random
import numpy as np
from tqdm import tqdm
from logger.logger_setup import setup_logger

# Logging setup
logger = setup_logger(name="TransferLearning", log_file="logs/transfer_learning.log")

# Seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration settings
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.conf')
config.read(config_path)

processed_data_path = config['data']['processed_data_path']
transfers_path = config['data']['transfers_path']
os.makedirs(transfers_path, exist_ok=True)

batch_size = int(config['finetune']['batch_size'])
num_epochs = int(config['finetune']['num_epochs'])
learning_rate = float(config['finetune']['learning_rate'])
valid_split = float(config['finetune']['valid_split'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Custom Dataset class
class ProcessedDataset(torch.utils.data.Dataset):
    '''
    A custom Dataset class for loading and processing images from a directory structure.
    Args:
        - root (str): Root directory path containing class subdirectories with images.
        - transform (callable, optional): A function/transform to apply to the images.
    Attributes:
        - root (str): Root directory path.
        - transform (callable, optional): A function/transform to apply to the images.
        - images (list): List of image file paths.
        - labels (list): List of labels corresponding to the images.
        - classes (list): List of class names.
        - class_to_idx (dict): Dictionary mapping class names to class indices.
    Methods:
        - __len__(): Returns the total number of images.
        - __getitem__(idx): Returns the image and label at the specified index.
    '''
    
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.images = []
        self.labels = []

        self.classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.classes.sort()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        for cls in self.classes:
            cls_path = os.path.join(root, cls)
            for video_dir in os.listdir(cls_path):
                video_path = os.path.join(cls_path, video_dir)
                if os.path.isdir(video_path):
                    for frame_name in os.listdir(video_path):
                        if frame_name.endswith('.jpg'):
                            frame_path = os.path.join(video_path, frame_name)
                            self.images.append(frame_path)
                            self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ProcessedDataset(root=processed_data_path, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
logger.info(f"Classes: {class_names} -- Total Classes: {num_classes}")
logger.info(f"Total frames in processed data: {len(dataset)}")

# Reduce dataset if necessary
max_samples = len(dataset)
if len(dataset) > max_samples:
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sampled_indices = indices[:max_samples]
    dataset = Subset(dataset, sampled_indices)
    logger.info(f"Dataset reduced to {len(dataset)} samples")

# Split dataset into training and validation sets
val_size = int(len(dataset) * valid_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model setup
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer to match the number of classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# Training and validation
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 5
counter = 0

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    total_train = 0

    for images, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total_train += images.size(0)

    epoch_loss = running_loss / total_train
    train_losses.append(epoch_loss)

    model.eval()
    val_running_loss = 0.0
    val_total = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            val_total += images.size(0)

    val_loss = val_running_loss / val_total
    val_losses.append(val_loss)

    logger.info(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}")

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        best_model_path = os.path.join(transfers_path, 'resnet_finetuned.pth')
        torch.save({'model_state_dict': model.state_dict(), 'class_names': class_names}, best_model_path)
        logger.info(f"Model saved at {best_model_path}")
    else:
        counter += 1
        logger.info(f"No improvement in validation for {counter} consecutive epochs.")
        if counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}.")
            break

if __name__ == "__main__":
    try:
        pass
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
