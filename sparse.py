import image_tools as img_tl
import numpy as np
import scipy as sp
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import math

from sklearn import linear_model
import sys


import matplotlib.pyplot as plt

lowPatchSize = (3,3)
highPatchSize = (6,6)
atoms = 200
lmbda = 1
iterations = 300

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
        img_height, img_width = img.size
        print(img.size)
        
        if img_height % 2 == 1:
            img = img.resize((img_height-1,img_width))
            img_height -= 1
        
        if img_width % 2 == 1:
            img = img.resize((img_height,img_width-1))
            img_width -= 1
        
        # ceil cause worried about 0 size
        img_low = img.resize((math.ceil(img_height*lowPatchSize[0]/highPatchSize[0]), math.ceil(img_width*lowPatchSize[1]/highPatchSize[1])))
        
        
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
        
        resultHigh = result[:, :highPatchSize[0]*highPatchSize[1]]
        resultLow = result[:, highPatchSize[0]*highPatchSize[1]:]
        
        np.save("models/sparseHigh.npy", resultHigh)
        np.save("models/sparseLow.npy", resultLow)
        
        inner_stats = trainer.inner_stats_

    print(result.shape)
    resultHigh = result[:, :highPatchSize[0]*highPatchSize[1]]
    resultLow = result[:, highPatchSize[0]*highPatchSize[1]:]
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
    return (temp - np.mean(temp, axis=1).reshape(temp.shape[0],1))


def super_size(image_paths):
    """Dh: high resolution dictionary
       Dl: low resolution dictionary
       Y:  low resolution image
    """
    # load model
    Dh = np.load("models/sparseHigh.npy")
    Dl = np.load("models/sparseLow.npy")
    
    print(Dh.shape)
    print(Dl.shape)
    
    # load image
    """ TEMP 1 image"""
    Y = np.array(img_tl.importImage(image_paths[0]))
    patches = img_tl.get_patches(Y[:,:]/255, lowPatchSize)
    
    """ temporarily only focus on one channel"""
    # prep the image data
    patches = patches.reshape(patches.shape[0], -1)
    means = np.mean(patches, axis=1).reshape(patches.shape[0],1)
    stdev = np.std(patches, axis=1).reshape(patches.shape[0],1)
    patches = (patches - means)
    
    # find the a representation
    Dlt = np.transpose(Dl)
    Dht = np.transpose(Dh)
    
    reconstructed_patches = np.zeros((patches.shape[0], highPatchSize[0]*highPatchSize[1]))
    
    for i in range(patches.shape[0]):
        if i%1000 == 0:
            print(".")
        
        patch = patches[i]
        
        trainer = Ridge(alpha = lmbda,
                        max_iter = 2)
        
        trainer.fit(Dlt, patch)
        
        reconstructed_patches[i] = Dht.dot(trainer.coef_.reshape(atoms,1)).reshape(highPatchSize[0]*highPatchSize[1])
    
    print(means.shape)
    reconstructed_patches += means
    reconstructed_patches = reconstructed_patches.reshape(img_tl.merge_tuples((patches.shape[0]), highPatchSize))
    
    patches = img_tl.get_patches(Y[:,:]/255, lowPatchSize)
    fixedImage, image2, image3 = img_tl.merge_patches(reconstructed_patches, (Y.shape[0]*2 ,Y.shape[1]*2), 2)
    fixedImage *= 255
    fixedImage = fixedImage.astype(np.uint8)
    img_tl.save_as_image(fixedImage, "fixed.png")
    
    image2 *=255
    image2 = image2.astype(np.uint8)
    img_tl.save_as_image(image2, "fixed2.png")
    image3 *=255
    image3 = image3.astype(np.uint8)
    img_tl.save_as_image(image3, "fixed3.png")
    
    
        
        
    