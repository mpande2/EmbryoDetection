# Convolution Neural Network-based classification of fertilized vs non-fertilized frog eggs

The project aims to develop a convolutional neural network to perform binary classification of frog eggs into fertilized (F) vs non-fertilized (NF) categories. 
The project aims to help experimental biologists determine
  - If the eggs from a particular frogs should be considered for experimental testing based on their viability ratio
  - To determine if the egg petri-dish is useful for further experimental testing

## Pre-requisites
- python 3
- Pillow `pip install pillow`
- tensorflow

### Generating synthetic dataset
In general a petridish with ~60% fertilized eggs and egg count ~300 is considered to be useful for experimental testing. So, we generated synthetic images with ~ 300 images with 60:40, 70:30, 80:30 and 90:10 fertilized vs non-fertilized eggs.
To generate synthetic images please use following code
```
# To genrate the synthetic dataset, please run RunScript.sh
# Generating ~300 eggs takes 3 seconds.
#User can adjust the number of eggs based on their computational power and radius of image
sh RunScript.sh
# The run script uses GenerateSyntheticImages_MultipleTypes.py script to generate images
#  and saves them into a folder synthetic_frog_eggs/mixture
```
