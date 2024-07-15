from ultralytics import YOLO
import os, sys
from clearml import Task 
import logging
import numpy as np 

OPTIMIZER = sys.argv[1]
# https://docs.ultralytics.com/tasks/detect/#why-should-i-use-ultralytics-yolov8-for-object-detection
# A-tavv's question for reference

# Set up logging
logging.basicConfig(filename='yolo_training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Load the ClearML paramters
task = Task.init(project_name="ImageDetectionProject", task_name="YOLOtrainingSyntheticData") 


# load a pretrained YOLOv8n model
model = YOLO("yolov10b.pt")  
logger.info("Loaded YOLO model from 'yolov10b.pt'")

# Log initial message
logger.info("Starting YOLO training script")

# Train the model
#for learningrate in np.linspace(0.001, 0.1, 3):
#for optimizer in ["SGD", "Adam", "AdamW", "NAdam", "RAdam"]:
# Train the model
for epoch in range(50, 200, 50):
	for IOU in np.linspace(0.1, 0.8, 8):
		logger.info(f"Starting training with Optimizer = Adam\n")

		try:
			results = model.train(
				data="data.yaml", 
				epochs=epoch, 
				imgsz=640,
				weight_decay=0.0005, 
				batch=16, 
				optimizer=OPTIMIZER, 
				seed=42, 
				dropout=0.5,
				conf=0.45,
				lr0=0.01,
				lrf=0.01,
				cos_lr=True, 
				iou=float(IOU), 
				flipud=0.5, 
				fliplr=0.5, 
				bgr=0.5, 
				mixup=0.5,
				mosaic=1.0, 
				#hsv_h=0.015, 
				#hsv_s=0.7, 
				#hsv_v=0.4,
				device='cuda:0,1', 
				amp=True, 
				project=f'SGD_training', 
				name=f'SGD_{epoch}_{IOU}', 
				val=True, 
				save=True, 
				plots=True, 
				verbose=True
			)
			logger.info("Training completed")
		except Exception as e:
			logger.error(f"An error occurred during training: {e}")
