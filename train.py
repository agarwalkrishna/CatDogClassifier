import pickle
import time
import numpy as predictClass
import h5py
import matplotlib.pyplot as plt
import scipy
from predict import *
from PIL import Image
from scipy import ndimage
from Cat_Dog_Classifier import *


plt.rcParams['figure.figsize'] = (7.0, 7.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, val_x_orig, val_y_orig, test_x_orig, test_y, classes = load_dataset()


m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print()
print()

print ("Number of training examples: {} ".format(m_train))
print ("Number of testing examples:  {} " .format(m_test))
print ("Each image is of size:       ({}, {}, 3)".format(num_px,num_px))
print()

print ("train_x shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_ shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

# Converting the shape from [m,num_px,num_px,num_px,3] to [num_px*num_px*num_px*3,m]
# Also known as flattening of dimentions
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
val_x_flatten = val_x_orig.reshape(val_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Normalize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.
val_x = val_x_flatten/255.
#print("train_x "+str(train_x[1:10][1]))
#print ("train_x's shape: " + str(train_x.shape))
#print ("test_x's shape: " + str(test_x.shape))


def model(X, Y, layers_dims, learning_rate = 0.001, lambd=0.1, mini_batch_size=32, num_iterations = 2000, print_cost=False):
    
    if(lambd>0):
        print("Regularization: YES")

    else:
        print("Regularization: NO")

    print("Regularization factor:       "+str(lambd))
    print("Number of Layers:            "+str(len(layers_dims)-1))
    print("Mini Batch Size:             "+str(mini_batch_size))
    print("Learning Rate:               "+str(learning_rate))
    print("Number of iterations:        "+str(num_iterations))
    print()
    print()

    costs = []                         
    
    parameters = initialize_parameters(layers_dims)
        
    for i in range(0, num_iterations):

        seed=3
        minibatches = generate_mini_batches(X, Y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            
            AL, caches = forward_propagation(minibatch_X, parameters)
            
            cost = compute_cost(AL, minibatch_Y, parameters, lambd)
            
            grads = backward_propagation(AL, minibatch_Y, caches, lambd)
            
            parameters = update_parameters(parameters, grads, learning_rate)
            
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    # Plot Cost v/s No. of iterations
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    fig1=plt.gcf()
    plt.show()
    fig1.savefig('results/Cost.png', bbox_inches='tight')
    
    return parameters

def main():
    
    layers_dims = [12288, 20, 10, 5, 1] #  4-layer model
    print("layers dimensions: "+str(layers_dims))
    parameters = model(train_x, train_y, layers_dims, lambd=0.01, learning_rate = 0.003, mini_batch_size=32, num_iterations = 2000, print_cost = True)

    fileObject=open('SavedParameters','wb')
    pickle.dump(parameters,fileObject)
    fileObject.close()

    print()
    pred_train,train_accuracy = predictClass(train_x, train_y)
    print("Train Accuracy: "+str(train_accuracy))
    pred_test,test_accuracy = predictClass(test_x, test_y)
    print("Test  Accuracy: "+str(test_accuracy))
    print_mislabeled_images(classes, test_x, test_y, pred_test)

if __name__=='__main__':
    main()
