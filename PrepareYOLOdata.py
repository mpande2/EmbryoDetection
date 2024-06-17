import os, sys
import random

# Before running this script cd to synthetic_frog_eggs
# cd synthetic_frog_eggs

# List all the png fles in the folder
images = [image for image in os.listdir() if ".png" in image]

# Select 80% of these images to be saved as training set
train_dir = "train"
if not os.path.isdir(train_dir):
    os.mkdir(train_dir)

train_images = "train/images"
train_label = "train/labels"

if not os.path.isdir(train_images):
    os.mkdir(train_images)

if not os.path.isdir(train_label):
    os.mkdir(train_label)

val_dir = "val"
if not os.path.isdir(val_dir):
    os.mkdir(val_dir)

val_images = "val/images"
val_label = "val/labels"

if not os.path.isdir(val_images):
    os.mkdir(val_images)

if not os.path.isdir(val_label):
    os.mkdir(val_label)

test_dir = "test"
if not os.path.isdir(test_dir):
    os.mkdir(test_dir)

test_images = "test/images"
test_label = "test/labels"

if not os.path.isdir(test_images):
    os.mkdir(test_images)

if not os.path.isdir(test_label):
    os.mkdir(test_label)

# Select 80% of the images as train
train_samples = random.sample(images, int(len(images)*0.8))

# Remaining images
images = [image for image in images if image not in train_samples]

# Select 50% as validation from remaining images
val_samples = random.sample(images, int(len(images)*0.5))

# Remaining images will be test data
test_samples = [image for image in images if image not in val_samples]

for image in train_samples:
  name = image.split(".")[0]
  os.system(f"mv {image} train/images")
  os.system(f"mv {name}.txt train/labels")

for image in val_samples:
  name = image.split(".")[0]
  os.system(f"mv {image} val/images")
  os.system(f"mv {name}.txt val/labels")

for image in test_samples:
  name = image.split(".")[0]
  os.system(f"mv {image} test/images")
  os.system(f"mv {name}.txt test/labels")
  
