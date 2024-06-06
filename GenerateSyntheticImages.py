from PIL import Image, ImageDraw
import numpy as np
import os
import random
import os, sys
import random 

# Load the reference images
viable_image_path = 'ExampleData/viable.png'  # Update with correct path if necessary ##### List of image names
non_viable_image_path = 'ExampleData/non-viable1.png'  # Update with correct path if necessary #### list of images
#random.sample(viable_image_path)


viable_image = Image.open(viable_image_path)
non_viable_image = Image.open(non_viable_image_path)

def is_overlapping(x, y, existing_positions, radius):
    for (ex, ey) in existing_positions:
        if np.sqrt((x - ex)**2 + (y - ey)**2) < 2 * radius:
            return True
    return False

# Function to create synthetic frog egg images
def create_synthetic_frog_egg_image(viable_img, non_viable_img, num_viable=50, num_non_viable=50, image_size=(500, 500)):
    synthetic_image = Image.new('RGB', image_size, (0, 0, 0))
    egg_radius = 20  # radius for synthetic eggs
    viable_img_resized = viable_img.resize((egg_radius * 2, egg_radius * 2))
    non_viable_img_resized = non_viable_img.resize((egg_radius * 2, egg_radius * 2))

    positions = []

    # Place viable eggs
    for _ in range(num_viable):
        while True:
            x = np.random.randint(egg_radius, image_size[0] - egg_radius)
            y = np.random.randint(egg_radius, image_size[1] - egg_radius)
            if not is_overlapping(x, y, positions, egg_radius):
                positions.append((x, y))
                synthetic_image.paste(viable_img_resized, (x - egg_radius, y - egg_radius))
                break

    # Place non-viable eggs
    for _ in range(num_non_viable):
        while True:
            x = np.random.randint(egg_radius, image_size[0] - egg_radius)
            y = np.random.randint(egg_radius, image_size[1] - egg_radius)
            if not is_overlapping(x, y, positions, egg_radius):
                positions.append((x, y))
                synthetic_image.paste(non_viable_img_resized, (x - egg_radius, y - egg_radius))
                break

    return synthetic_image

# Create directories for synthetic dataset
os.makedirs('synthetic_frog_eggs/mixture', exist_ok=True)

# Generate synthetic images
num_images = 1000
for i in range(num_images):
    # Create image with both viable and non-viable eggs
    # 10:20, 20:30, 20:10 
    synthetic_image = create_synthetic_frog_egg_image(viable_image, non_viable_image, num_viable=int(sys.argv[1]), num_non_viable=int(sys.argv[2]))
    synthetic_image.save(f'synthetic_frog_eggs/mixture/synthetic_frog_egg_{i}.png')

print('Synthetic frog egg images generated successfully.')

