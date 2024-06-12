import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths
TRAIN_DATA_DIR = 'synthetic_frog_eggs'
IMG_DIR = os.path.join(TRAIN_DATA_DIR, 'images')
LABEL_DIR = os.path.join(TRAIN_DATA_DIR, 'labels')
PROCESSED_DATA_DIR = 'synthetic_frog_eggs/processed'
IMG_SIZE = (256, 256)


# Create processed data directory
if not os.path.exists(PROCESSED_DATA_DIR):
    os.makedirs(PROCESSED_DATA_DIR)

# Function to parse YOLO annotations
def parse_yolo_annotation(txt_file, img_width, img_height):
    bboxes = []
    with open(txt_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center = int(x_center * img_width)
            y_center = int(y_center * img_height)
            radius = int(min(width * img_width, height * img_height) / 2)
            bboxes.append((class_id, x_center, y_center, radius))
    return bboxes

# Function to process images and annotations
def process_images():
    for file_name in os.listdir(IMG_DIR):
        if file_name.endswith('.png'):
            img_path = os.path.join(IMG_DIR, file_name)
            txt_path = os.path.join(LABEL_DIR, file_name.replace('.png', '.txt'))

            # Read and resize image
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]
            img_resized = cv2.resize(img, IMG_SIZE)
            
            # Parse annotation
            bboxes = parse_yolo_annotation(txt_path, img_width, img_height)

            # Create label mask for each bounding box
            label = np.zeros(IMG_SIZE, dtype=np.uint8)
            for bbox in bboxes:
                class_id, x_center, y_center, radius = bbox
                x_center = int(x_center * IMG_SIZE[0] / img_width)
                y_center = int(y_center * IMG_SIZE[1] / img_height)
                radius = int(radius * min(IMG_SIZE[0], IMG_SIZE[1]) / min(img_width, img_height))
                cv2.circle(label, (x_center, y_center), radius, (255 if class_id == 1 else 128), -1)

            # Save processed images and labels
            cv2.imwrite(os.path.join(PROCESSED_DATA_DIR, file_name), img_resized)
            cv2.imwrite(os.path.join(PROCESSED_DATA_DIR, file_name.replace('.png', '_label.png')), label)

process_images()
