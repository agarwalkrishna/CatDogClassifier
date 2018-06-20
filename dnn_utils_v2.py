import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

def sigmoid(Z):
       
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backprop(dA, cache):
   
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backprop(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def read_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def load_all_data():
    
    dataset = h5py.File("dataset.hdf5", "r")

    #print("train_set_x "+str(dataset["train_set_x"][:]))
    train_set_x_orig = np.array(dataset["train_set_x"][:])
    train_set_y_orig = np.array(dataset["train_set_y"][:])

    dev_set_x_orig = np.array(dataset["val_set_x"][:])
    dev_set_y_orig = np.array(dataset["val_set_y"][:])

    test_set_x_orig = np.array(dataset["test_set_x"][:])
    test_set_y_orig = np.array(dataset["test_set_y"][:])

    train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0], 1))
    dev_set_y_orig = dev_set_y_orig.reshape((dev_set_y_orig.shape[0], 1))
    test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0], 1))
    classes={"0":"Cat","1":"Dog"}
    #classes = dataset["list_classes"][:]
    
    return train_set_x_orig, train_set_y_orig, dev_set_x_orig, dev_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    #print("m "+str(m))
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    #print("X shape "+str(X.shape))
    #print("Y.shape "+str(Y.shape))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y.reshape((1,m))[:, permutation]
    
    

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    #print("num_complete_minibatches "+str(num_complete_minibatches))
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1) * mini_batch_size]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
   
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[:, (k+1) * mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:, (k+1) * mini_batch_size:m]
        ### END CODE HERE ###
        
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches

def print_mislabeled_images(classes, X, y, p):
    y=y.T
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    print("Number of Test Images: "+str(y.shape[1]))
    print("Number of mislabeled images: "+str(num_images))
    '''for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        #plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))'''

        
