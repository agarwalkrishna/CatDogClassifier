import numpy as np
import h5py
import math
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def sigmoid(Z):
       
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    
    A = np.maximum(0,Z)
    cache = Z 
    
    assert(A.shape == Z.shape)    

    return A, cache


def relu_backprop(dA, cache):
   
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backprop(dA, cache):
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def load_dataset():
    
    dataset = h5py.File("dataset.hdf5", "r")

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
    
    return train_set_x_orig, train_set_y_orig, dev_set_x_orig, dev_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def generate_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    #Creates a list of random minibatches from (X, Y)
    
    np.random.seed(seed)            
    m = X.shape[1]                  
    mini_batches = []

    # Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y.reshape((1,m))[:, permutation]

    # Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k*mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k*mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # If (last mini-batch < mini_batch_size)
   
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, (k+1) * mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:, (k+1) * mini_batch_size:m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches    

def initialize_parameters(layer_dims):
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) * np.sqrt(2./layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros(shape=(layer_dims[l],1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters


def forward_activation(A_prev, W, b, activation):

    Z = W.dot(A_prev) + b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    linear_cache = (A_prev, W, b)

    if activation == "sigmoid":

        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
       
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters):

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        W=parameters["W"+str(l)]
        b=parameters["b"+str(l)]
       
        A, cache = forward_activation(A_prev,W,b,"relu")
        
        caches.append(cache)
    
    AL, cache = forward_activation(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
   
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches



def compute_cost(AL, minibatch_Y, parameters, lambd):
    
    m = minibatch_Y.shape[1]
    L=len(parameters)//2
    logprobs = (-np.dot(minibatch_Y,np.log(AL).T) - np.dot(1-minibatch_Y, np.log(1-AL).T))
    l2_cost = 1./m * np.nansum(logprobs)
    
    regularization_cost=0

    for l in range(1, L+1):
        W=parameters["W"+str(l)]
        b=parameters["b"+str(l)]
        regularization_cost =regularization_cost+ np.sum(np.sum(np.square(W)))

    regularization_cost=(1./(2*m))*lambd*regularization_cost
    cost=l2_cost+regularization_cost
    cost = np.squeeze(cost)      # This converts [[20]] into 20).
    assert(cost.shape == ())
    
    return cost

def linear_activation_backward(dA, cache, activation, lambd):
   
    linear_cache, activation_cache = cache
    
    if activation == "sigmoid":

        dZ = sigmoid_backprop(dA,activation_cache)
        
    elif activation == "relu":

        dZ = relu_backprop(dA,activation_cache)
        
    A_prev, W, b = linear_cache
    m = A_prev.shape[1]

    
    dW = 1./m * np.dot(dZ,A_prev.T)+(lambd*W)/m
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    return dA_prev, dW, db


def backward_propagation(AL, Y, caches, lambd):
    
    grads = {}
    L = len(caches)         #Number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) #This ensures that the shape of Y and AL are same
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache =caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,activation="sigmoid",lambd=lambd)
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,activation="relu",lambd=lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2 # Number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l+1)]
    
    return parameters


def get_class_name(label):

    classes={"0":"Cat","1":"Dog"}
    return classes[str(label)]


def print_mislabeled_images(classes, X, y, p):

    y=y.T
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    num_images = len(mislabeled_indices[0])

    print("Number of Test Images: "+str(y.shape[1]))
    print("Number of mislabeled images: "+str(num_images))

        