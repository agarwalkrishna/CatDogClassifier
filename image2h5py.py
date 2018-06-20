from random import shuffle
import glob
import numpy as np
import h5py
from scipy import ndimage,misc
from skimage import transform

import warnings
warnings.filterwarnings("ignore")




shuffle_data = True                  # Shuffle the addresses before saving
hdf5_path = 'dataset.hdf5'           # Address to where you want to save the hdf5 file

''' Give the path to the folder where your images are stored 
    In case your folder contains other subflders then give cat_dog_train_path = 'dataset/*/*.jpg' 
'''
cat_dog_train_path = 'dataset/*.jpg' 


def create_h5py_dataset(num_pix=64,ratio=0.30):

    addrs = glob.glob(cat_dog_train_path)
    labels = [0 if 'cat' in addr else 1 for addr in addrs]  
    
    if shuffle_data:
        c = list(zip(addrs, labels))
        shuffle(c)
        addrs, labels = zip(*c)
    print("Total no.of training images: "+str(len(addrs)))
    addrs=addrs[0:int(ratio*len(addrs))]
    labels= labels[0:int(ratio*len(labels))]   

    # Split the Data - 60% train, 20% validation, and 20% test.
    train_addrs = addrs[0:int(0.6*len(addrs))]
    train_labels = labels[0:int(0.6*len(labels))]
    val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
    val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
    test_addrs = addrs[int(0.8*len(addrs)):]
    test_labels = labels[int(0.8*len(labels)):]
    
    train_shape = (len(train_addrs), num_pix, num_pix, 3)
    val_shape = (len(val_addrs), num_pix, num_pix, 3)
    test_shape = (len(test_addrs), num_pix, num_pix, 3)
    
    # Open a hdf5 file and create earrays
    hdf5_file = h5py.File(hdf5_path, mode='w')

    hdf5_file.create_dataset("train_set_x", train_shape, np.uint8)
    hdf5_file.create_dataset("val_set_x", val_shape, np.uint8)
    hdf5_file.create_dataset("test_set_x", test_shape, np.uint8)
   
    hdf5_file.create_dataset("train_set_y", (len(train_addrs),), np.uint8)
    hdf5_file["train_set_y"][...] = train_labels
    hdf5_file.create_dataset("val_set_y", (len(val_addrs),), np.uint8)
    hdf5_file["val_set_y"][...] = val_labels
    hdf5_file.create_dataset("test_set_y", (len(test_addrs),), np.uint8)
    hdf5_file["test_set_y"][...] = test_labels
    all_addrs=(train_addrs,val_addrs,test_addrs)
    
    train_addrs,val_addrs,test_addrs=all_addrs
    
    # Creating train set
    for i in range(len(train_addrs)):
        # Print how many images are saved every 1000 images

        addr = train_addrs[i]
        if i % 100 == 0 and i > 1:
            print('Train data: {}/{}'.format(i, len(train_addrs)))
            
        img = ndimage.imread(addr, flatten=False).astype(np.float)
        img = transform.resize(img, (num_pix, num_pix,3))
       
        hdf5_file["train_set_x"][i, ...] = img

    # Creating validation set
    for i in range(len(val_addrs)):
        # Print how many images are saved every 1000 images
        if i % 100 == 0 and i > 1:
            print('Validation data: {}/{}'.format(i, len(val_addrs)))

        addr = val_addrs[i]
        img = ndimage.imread(addr, flatten=False).astype(np.float)
        img = transform.resize(img, (num_pix, num_pix,3))

        hdf5_file["val_set_x"][i, ...] = img

    # Creating test set
    for i in range(len(test_addrs)):
        # Print how many images are saved every 1000 images
        if i % 100 == 0 and i > 1:
            print('Test data: {}/{}'.format(i, len(test_addrs)))

        addr = test_addrs[i]
        img = ndimage.imread(addr, flatten=False).astype(np.float)
        img = transform.resize(img, (num_pix, num_pix,3))

        hdf5_file["test_set_x"][i, ...] = img

    # Save and close the hdf5 file
    hdf5_file.close()


if __name__ == '__main__':

    # num_pix = The number of pixels you want to reshape your image to.
    # ratio   = The part of the dataset you want to work on.
    create_h5py_dataset(num_pix=64,ratio=0.30)
