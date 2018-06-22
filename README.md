# Binary Classifier to differentiate between Cats and Dogs.

# Requirements:
Anconda

# Components:

1. image2h5py.py - A python script to convert the image dataset into ndarrays and store them in h5py format. This not only saves disk space of storing the entire dataset but also makes it faster to retrive the dataset while training.

2. Cat_Dog_Classifier.py - Contains the various utility functions for Forward and Backward Propagation.

3. train.py - The main training model is described in this file which makes use of the utility functions defined in Cat_Dog_Classifier.py.

4. SavedParameters- This is a pickle file which stores the updated hidden layer parameters (weights and biases) of the neural network after training. These parameters are used while prediction.

5. predict.py - Script to predict the class (0:Cat, 1: Dog) of a test image given the image path.
