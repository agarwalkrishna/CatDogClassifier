# Binary Classifier to differentiate between Cats and Dogs.

This project implements a 4-layered Neural Network to build a classifier that can differenciate between images of cats and dogs. The dataset has been taken from [Kaggle's](https://www.kaggle.com/c/dogs-vs-cats/data) famous 'Dogs vs Cats' competion which contains 25,000 images of cats and dogs.

## Requirements:
Anconda

## Components:

1. image2h5py.py - A python script to convert the image dataset into ndarrays and store them in h5py format. This not only saves disk space of storing the entire dataset but also makes it faster to retrive the dataset while training.

2. Cat_Dog_Classifier.py - Contains the various utility functions for Forward and Backward Propagation.

3. train.py - The main training model is described in this file which makes use of the utility functions defined in Cat_Dog_Classifier.py.

4. SavedParameters- This is a pickle file which stores the updated hidden layer parameters (weights and biases) of the neural network after training. These parameters are used while prediction.

5. predict.py - Script to predict the class (0: Cat, 1: Dog) of a test image given the image path.

## Steps to Train:

1. Download and configure Anaconda for python3.x from [here](https://www.anaconda.com/download/).

2. Clone the [project](https://github.com/agarwalkrishna/CatDogClassifier.git).

3. Download the 'train.zip' file from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) and unzip it into the CatDogClassifier folder.

4. Run the 'image2h5py.py' file from command line to convert the images into an h5py dataset.

5. Run the train.py file to train on the dataset. This may take anywhere from a few minutes to an hour depending on your system configurations (mainly RAM, GPU and Processor).

6. After training the trained parameters will be stored in the 'SavedParameters' pickle file.

## Steps to Test:

You can train the model by passing your own value of hyperparameters or can simply test the model on different images without any prior training. The 'SavedParameters' file contains parameters from previously trained model.

1. Run the predict.py and pass the path to your test image from command line. Suppose you folder structure like:
            
            -CatDogClassifier
              -testImages
                -test1.jpg            
Simply write the below in command line -
          ```
            python3 testImages\test1.jpg
          ```         
The algorithm will predict and return the class (Cat or Dog) name of your test image.
