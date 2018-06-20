import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import transform
from Cat_Dog_Classifier import forward_propagation,get_class_name

def predictClass(X, y):
     
    fileObject=open('SavedParameters','rb')

    #loading the saved parameters obtained from previously trained dataset    
    parameters=pickle.load(fileObject)
    m = X.shape[1]
    
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    try:
        y=y.T
    except Exception:
        y=y
    
    probas, caches = forward_propagation(X, parameters)

    #Threshold value is set to be 0.5
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    accuracy=(np.sum(p == y))/m
    return p,accuracy

if __name__=='__main__':
    
    my_image = sys.argv[1] # Image path- input from command line
    y= sys.argv[2]         # the true class of the input image (0 -> Cat, 1 -> Dog)
    
    num_px=64
    fname =  my_image
    image = ndimage.imread(fname, flatten=False).astype(np.float)
    my_image = transform.resize(image, (64, 64,3))
    my_image=my_image.reshape(num_px*num_px*3,1)
    
    X = my_image/255.
    my_predicted_image,accuracy = predictClass(X, y)

    plt.imshow(image)

     
    img_class=get_class_name(int(np.squeeze(my_predicted_image)))
    print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + img_class +  "\" picture.")