#!/bin/bash

# Loop through each image in the TestImages directory with specified extensions
for image in ProcessedImages/TestImages2/*.{jpg,png,tif,tiff}; do
    # Check if the file exists (in case there are no matches for certain extensions)
    if [ -e "$image" ]; then
        # Extract the name without extension and directory
        name=$(basename "$image")
        name="${name%.*}"  # Remove the extension

        # Check if the corresponding text file exists
        if [ -e "ProcessedImages/predict/labels/$name.txt" ]; then
            # Run the bounding box script
            echo "Processing $image with bounding box data from ProcessedImages/predict/labels/$name.txt"
            python3.11 boundingbox.py "$image" "ProcessedImages/predict/labels/$name.txt"
        else
            echo "No bounding box file found for $image. Skipping..."
        fi
    fi
done
