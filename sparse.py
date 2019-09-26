import image_tools as img_tl
import numpy as np
import scipy as sp
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import math

from sklearn import linear_model
import sys


import matplotlib.pyplot as plt

lowPatchSize = (3,3)
highPatchSize = (6,6)
representationSize = (3,3)
atoms = 100
lmbda = 1
iterations = 100

def train(image_paths):

    inner_stats = None

    # train on each image
    
    for image_path in image_paths:
    
        # import image
        try:
            img = img_tl.importImage(image_path)
        except IOError:
            print("Error, an image could not be found in the images directory. Did you move it while the machine was training?")
            sys.exit()
        
        # prepare images to high and low res patches
        img_x, img_y = img.size
        
        if img_x % 2 == 1:
            img = img.resize((img_x-1,img_y))
            img_x -= 1
        
        if img_y % 2 == 1:
            img = img.resize((img_x,img_y-1))
            img_y -= 1
        
        # ceil cause worried about 0 size
        img_low = img.resize((math.ceil(img_x*0.5), math.ceil(img_y*0.5)))
        
        
        # convert to patches
        img = np.array(img)
        img_low = np.array(img_low)
        
        """ for now use only red channel """
        img = img[:,:,0]
        img_low = img_low[:,:,0]
        
        # print out the image for later testing
        img_tl.save_as_image(img, "red.png")
        img_tl.save_as_image(img_low, "red_low.png")
        """-----------------------------"""
        
        print(img.shape)
        print(img_low.shape)
        
        high_data = convertImageDataToPatches(img, highPatchSize, 2)
        low_data = convertImageDataToPatches(img_low, lowPatchSize)
        
        high_data_size = high_data.shape[1]
        low_data_size = low_data.shape[1]
        
        high_data *= 1/math.sqrt(high_data_size)
        low_data *= 1/math.sqrt(low_data_size)
        
        print(high_data.shape)
        print(low_data.shape)
        
        data = np.concatenate((high_data, low_data), axis = 1)
        print(data.shape)
    
        trainer = MiniBatchDictionaryLearning(
        n_components = atoms,
        alpha = lmbda*(1/high_data_size + 1/low_data_size),
        n_iter = iterations,
        n_jobs = -1,
        verbose = 1)
        if inner_stats:
            trainer.inner_stats_ = inner_stats
    
        result = trainer.fit(data).components_
        
        np.save("models/sparse.npy", result)
        
        inner_stats = trainer.inner_stats_

    print(result.shape)
    resultHigh = result[:, :36]
    resultLow = result[:, 36:]
    plt.figure(figsize=(4.2, 4))
    for i, comp in enumerate(resultHigh[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(highPatchSize), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()
    for i, comp in enumerate(resultLow[:100]):
        plt.subplot(10, 10, i + 1)
        plt.imshow(comp.reshape(lowPatchSize), cmap=plt.cm.gray_r,
                   interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()
    
    
def convertImageDataToPatches(img, patchSize, stride=1):
    """ Converts image to a machine learning friendly format
        Does this by converting 256 bit colour to a scale of 0 to 1,
         turns it into patches with mean subtracted, and divides by std. dev
    """
    
    temp = img_tl.get_patches(img/255, patchSize, stride)
    temp = temp.reshape(temp.shape[0], -1)
    return (temp - np.mean(temp, axis=0))/np.std(temp,axis=0)


def super_size(Dh, Dl, Y):
    """Dh: high resolution dictionary
       Dl: low resolution dictionary
       Y:  low resolution image
    """
    