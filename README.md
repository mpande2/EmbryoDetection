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
# Here is the example of run to create ~300 eggs with 60:40 F vs NF ratio
python GenerateSyntheticImages.py 180 120
```
