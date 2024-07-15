import os
import sys
import numpy as np
from ultralytics import YOLO  # Assuming ultralytics is the correct package for YOLO

# Load a pretrained YOLOv8n model
model = YOLO("./model/best.pt")

# Log initial message

def non_max_suppression(boxes, scores, iou_threshold=0.5, class_agnostic=False, class_labels=None):
    sorted_indices = np.argsort(scores)[::-1]
    sorted_boxes = boxes[sorted_indices]
    selected_boxes = []
    selected_indices = []
    
    while len(sorted_boxes) > 0:
        current_box = sorted_boxes[0]
        selected_boxes.append(current_box)
        selected_indices.append(sorted_indices[0])
        rest_boxes = sorted_boxes[1:]
        ious = compute_iou(current_box, rest_boxes)
        
        if class_agnostic:
            sorted_boxes = rest_boxes[ious <= iou_threshold]
            sorted_indices = sorted_indices[1:][ious <= iou_threshold]
        else:
            current_class = class_labels[sorted_indices[0]]
            class_filter = class_labels[sorted_indices[1:]] == current_class
            sorted_boxes = rest_boxes[ious <= iou_threshold | ~class_filter]
            sorted_indices = sorted_indices[1:][ious <= iou_threshold | ~class_filter]
    
    return np.array(selected_boxes), selected_indices

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union_area = box_area + boxes_area - inter_area
    iou = inter_area / union_area
    
    return iou

def convert_to_corners(pred):
    # Convert from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    x_center, y_center, width, height = pred[:, 1], pred[:, 2], pred[:, 3], pred[:, 4]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def convert_to_yolo_format(pred, selected_boxes, selected_indices):
    # Create a new array for the selected predictions
    selected_pred = np.zeros((len(selected_boxes), pred.shape[1]))
    selected_pred[:, 0] = pred[selected_indices, 0]  # class
    selected_pred[:, 5] = pred[selected_indices, 5]  # confidence

    # Convert from [x1, y1, x2, y2] back to [x_center, y_center, width, height]
    x1, y1, x2, y2 = selected_boxes[:, 0], selected_boxes[:, 1], selected_boxes[:, 2], selected_boxes[:, 3]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    selected_pred[:, 1:5] = np.stack([x_center, y_center, width, height], axis=1)

    return selected_pred

# Directory to save all predictions
prediction_dir = "ProcessedImages"
os.makedirs(prediction_dir, exist_ok=True)

# Perform prediction
for image in os.listdir(f"{prediction_dir}/TestImages2/"):
    image_path = f"{prediction_dir}/TestImages2/{image}"
    name = os.path.splitext(image)[0]
    validation_results = model.predict(source=image_path, classes=[0,1], project=prediction_dir, agnostic_nms=True, conf=0.25, max_det=500, save=False, save_txt=True, save_conf=True, show_labels=False, show_conf=False, show_boxes=True, line_width=1, exist_ok=True)

    # Apply NMS
    # Load the predicted labels from YOLO
    pred_path = f"{prediction_dir}/predict/labels/{name}.txt"
    if not os.path.exists(pred_path):
        continue  # Skip if the label file doesn't exist
    
    pred = np.loadtxt(pred_path)
    if pred.ndim == 1:
        pred = np.expand_dims(pred, axis=0)  # Ensure pred is 2D for a single prediction case
    
    if len(pred) == 0:
        continue  # Skip if there are no predictions
    
    class_labels = pred[:, 0]
    boxes = convert_to_corners(pred)  # Convert to corner format for NMS
    scores = pred[:, 5]  # Confidence scores are at index 5
    selected_boxes, selected_indices = non_max_suppression(boxes, scores, iou_threshold=0.5, class_agnostic=True, class_labels=class_labels)
    pred = convert_to_yolo_format(pred, selected_boxes, selected_indices)  # Convert back to YOLO format
    #print(pred)

    # Save the filtered predictions
    np.savetxt(pred_path, pred, fmt='%f')
