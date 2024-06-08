from PIL import Image, ImageDraw
import numpy as np
import os
import sys
import random


# Load the reference images from directories
viable_image_dir = 'fertilized_eggs'  # Update with correct path if necessary
non_viable_image_dir = 'unfertilized_eggs'  # Update with correct path if necessary

#opens full path of an image for each img in directory
viable_images = [Image.open(os.path.join(viable_image_dir, img)) for img in os.listdir(viable_image_dir) if img.endswith('png')]
non_viable_images = [Image.open(os.path.join(non_viable_image_dir, img)) for img in os.listdir(non_viable_image_dir) if img.endswith('png')]


def is_overlapping(x, y, existing_positions, radius):
    for (ex, ey) in existing_positions:
        if np.sqrt((x - ex)**2 + (y - ey)**2) < 2 * radius:
            return True
    return False

def is_within_circle(x, y, center, radius):
    return np.sqrt((x - center[0])**2 + (y - center[1])**2) <= radius


# Function to create synthetic frog egg images
def create_synthetic_frog_egg_image(viable_imgs, non_viable_imgs, num_viable=50, num_non_viable=50, image_size=(1000, 1000), output_dir='dataset'):
    synthetic_image = Image.new('RGBA', image_size, (160, 172, 185, 255))  # light grey
    egg_radius = 20  # radius for synthetic eggs
    positions = []
    center = (image_size[0] // 2, image_size[1] // 2)
    circle_radius = 500  # radius 500 for a diameter of 1000
    margin = egg_radius + 4  # margin from the perimeter
    annotations = []

    # Place viable eggs
    for _ in range(num_viable):
        while True:
            x = np.random.randint(margin, image_size[0] - margin)
            y = np.random.randint(margin, image_size[1] - margin)
            if not is_overlapping(x, y, positions, egg_radius) and is_within_circle(x, y, center, circle_radius - margin):
                positions.append((x, y))
                viable_img = random.choice(viable_imgs).resize((egg_radius * 2, egg_radius * 2)).convert("RGBA")
                synthetic_image.paste(viable_img, (x - egg_radius, y - egg_radius), viable_img)
                # Add bounding box annotation (class 0 for viable)
                annotations.append((0, x, y, egg_radius))
                break
                
    # Place non-viable eggs
    for _ in range(num_non_viable):
        while True:
            x = np.random.randint(margin, image_size[0] - margin)
            y = np.random.randint(margin, image_size[1] - margin)
            if not is_overlapping(x, y, positions, egg_radius) and is_within_circle(x, y, center, circle_radius - margin):
                positions.append((x, y))
                non_viable_img = random.choice(non_viable_imgs).resize((egg_radius * 2, egg_radius * 2)).convert("RGBA")
                synthetic_image.paste(non_viable_img, (x - egg_radius, y - egg_radius), non_viable_img)
                # Add bounding box annotation (class 0 for viable)
                annotations.append((1, x, y, egg_radius))
                break

    # Create a circular mask with a grey circle and a black rim
    grey_circle = Image.new('RGBA', image_size, (0, 0, 0, 255))  # black
    grey_circle_mask = Image.new('L', image_size, 0)
    draw = ImageDraw.Draw(grey_circle_mask)
    draw.ellipse((center[0] - circle_radius, center[1] - circle_radius,
                  center[0] + circle_radius, center[1] + circle_radius), fill=255)
    grey_circle.paste(synthetic_image, (0, 0), grey_circle_mask)

    # Create the black rim
    draw = ImageDraw.Draw(grey_circle)
    draw.ellipse((center[0] - circle_radius, center[1] - circle_radius,
                  center[0] + circle_radius, center[1] + circle_radius), outline=(211, 211, 211, 255))  # light grey

    return grey_circle, annotations


# Function to save annotations in YOLO format
def save_yolo_annotations(annotations, output_path, image_size):
    with open(output_path, 'w') as f:
        for annotation in annotations:
            class_id, x, y, radius = annotation
            x_center = x / image_size[0]
            y_center = y / image_size[1]
            width = (2 * radius) / image_size[0]
            height = (2 * radius) / image_size[1]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


# Create directories for synthetic dataset
output_dir = 'synthetic_frog_eggs'
os.makedirs(output_dir, exist_ok=True)
#os.makedirs('synthetic_frog_eggs/mixture', exist_ok=True)
num_images = 20

for i in range(num_images):
    # Create image with both viable and non-viable eggs
    synthetic_image, annotations = create_synthetic_frog_egg_image(viable_images, non_viable_images, num_viable=int(sys.argv[1]), num_non_viable=int(sys.argv[2]), output_dir=output_dir)
    image_path = os.path.join(output_dir, f'synthetic_frog_egg_{sys.argv[1]}_{sys.argv[2]}_{i}.png')
    annotation_path = os.path.join(output_dir.replace('images', 'labels'), f'synthetic_frog_egg_{sys.argv[1]}_{sys.argv[2]}_{i}.txt')
    synthetic_image.save(image_path)
    save_yolo_annotations(annotations, annotation_path, synthetic_image.size)

print('Synthetic frog egg images and annotations generated successfully.')
