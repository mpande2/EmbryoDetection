import cv2
import matplotlib.pyplot as plt
import os, sys

# Create a prediction folder
pred_image_path = "ProcessedImages/predict/PredImages"
os.makedirs(pred_image_path, exist_ok=True)

# Load the image
image_path = sys.argv[1]
name = os.path.splitext(image_path)[0]
name = name.split("/")[-1]
print(name)
image = cv2.imread(image_path)
image_height, image_width, _ = image.shape


# Load and parse the text file
coordinates_path = sys.argv[2]
with open(coordinates_path, 'r') as file:
    lines = file.readlines()

# Draw bounding boxes
for line in lines:
    label, x_center, y_center, width, height, confidence = map(float, line.strip().split())
    # Convert normalized coordinates to actual coordinates
    x_center *= image_width
    y_center *= image_height
    box_width = width * image_width
    box_height = height * image_height
		# Calculate the top-left corner of the bounding box
    top_left_x = int(x_center - box_width / 2)
    top_left_y = int(y_center - box_height / 2)
    bottom_right_x = int(x_center + box_width / 2)
    bottom_right_y = int(y_center + box_height / 2)
    # Draw the bounding box
    color = (155, 255, 0) if label == 0 else (255, 0, 255)  # Green for label Fertilized, Purple for label unfertilized
    cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 5)

# Convert BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with bounding boxes
plt.figure(figsize=(20, 20))
plt.imshow(image_rgb)
plt.axis('off')
plt.tight_layout()

plt.savefig(f"{pred_image_path}/{name}_prediction.jpg", dpi=300)
