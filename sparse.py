import image_tools as img_tl
import numpy as np
import scipy as sp
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import math
import os

from sklearn import linear_model
import sys


import matplotlib.pyplot as plt

lowPatchSize = (3,3)
highPatchSize = (6,6)
atoms = 200
lmbda = 0.25
lmbda2= 1
iterations = 300

def train(image_paths):
    
    results = np.zeros((atoms, lowPatchSize[0]*lowPatchSize[1]+highPatchSize[0]*highPatchSize[1]))
    init_dict = True
    
    for image_path in image_paths:
        # import image
        try:
            print(image_path)
            img = img_tl.importImage(image_path)
        except IOError:
            print("Error, an image could not be found in the images directory. Did you move it while the machine was training?")
            sys.exit()
        
        img_height, img_width = img.size
        img_low,img = halfImageResolutionForTraining(img)
        img = np.array(img)
        img_low = np.array(img_low)
        
        
        
        # for each colour channel in the image
        """ see if we can just train and make only 1 model."""
        for channel in range(img.shape[2]):
        
            # convert to patches
            data_high = img[:,:,channel]
            data_low = img_low[:,:,channel]
            high_data = convertImageDataToPatches(data_high, highPatchSize, 2)
            low_data = convertImageDataToPatches(data_low, lowPatchSize)
            
            high_data_size = high_data.shape[1]
            low_data_size = low_data.shape[1]
            
            # mathematically reduce values to fit algorithm
            high_data *= 1/math.sqrt(high_data_size)
            low_data *= 1/math.sqrt(low_data_size)
            
            # join the high and low res data
            data = np.concatenate((high_data, low_data), axis = 1)
            
            # train
            trainer = None
            if(init_dict):
                trainer = MiniBatchDictionaryLearning(
                n_components = atoms,
                alpha = lmbda*(1/high_data_size + 1/low_data_size),
                n_iter = iterations,
                n_jobs = -1,
                verbose = 1)
            else:
                trainer = MiniBatchDictionaryLearning(
                n_components = atoms,
                alpha = lmbda*(1/high_data_size + 1/low_data_size),
                n_iter = iterations,
                n_jobs = -1,
                verbose = 1,
                dict_init = results)
                
            model = trainer.fit(data).components_
            
            results = model
        
        init_dict = False
    
    # save the result
    resultHigh = results[:, :highPatchSize[0]*highPatchSize[1]]
    resultLow = results[:, highPatchSize[0]*highPatchSize[1]:]
    np.save("models/sparseHigh.npy", resultHigh)
    np.save("models/sparseLow.npy", resultLow)
    
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
    
def halfImageResolutionForTraining(img):
    # prepare images to high and low res patches
    img_height, img_width = img.size
    
    if img_height % 2 == 1:
        img = img.resize((img_height-1,img_width))
        img_height -= 1
    
    if img_width % 2 == 1:
        img = img.resize((img_height,img_width-1))
        img_width -= 1
    
    # ceil cause worried about 0 size
    img_low = img.resize(
        (math.ceil(img_height*lowPatchSize[0]/highPatchSize[0]), 
        math.ceil(img_width*lowPatchSize[1]/highPatchSize[1])))
        
    return img_low,img

    
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
    np.warnings.filterwarnings('ignore')
    
    # load model
    Dh = np.load("models/sparseHigh.npy")
    Dl = np.load("models/sparseLow.npy")
    
    Dht = np.transpose(Dh)
    
    for image_path in image_paths:
    
        # get image data and a bit of prep
        Y = np.array(img_tl.importImage(image_path))
        height, width, channels = Y.shape
        patches = img_tl.get_patches(Y[:,:]/255, lowPatchSize)
        
        reconstruction = np.zeros((height*2, width*2, channels)).astype(np.uint8)
        
        for channel in range(patches.shape[-1]):
            
            # prep patches for fitting
            ch_patches = np.copy(patches[:,:,:,channel])
            ch_patches = ch_patches.reshape(patches.shape[0], -1)
            means = np.mean(ch_patches, axis=1).reshape(ch_patches.shape[0],1)
            ch_patches = ch_patches - means
            
            
            reconstructed_patches = np.zeros((patches.shape[0], highPatchSize[0]*highPatchSize[1]))
            reconstructed_image = np.zeros((height*2 ,width*2))
            reconstructed_multiples = np.zeros((height*2 ,width*2))
            
            trainer = Ridge(alpha = lmbda2,
                    max_iter = iterations)
            stride = highPatchSize[0]//lowPatchSize[0]
            
            for i in range(0,height*2-highPatchSize[0]+1, stride):
                if i%100 == 0:
                    print(".")
                for j in range(0, width*2-highPatchSize[1]+1, stride):
                    
                    patch_i = i//stride*((width*2 - highPatchSize[1])//stride + 1) + j//stride
                    patch = ch_patches[patch_i]
                    
                    prev_overlap, mask = GenOverlapMask(
                        reconstructed_image[i:i+highPatchSize[0], j:j+highPatchSize[1]],
                        reconstructed_multiples[i:i+highPatchSize[0], j:j+highPatchSize[1]],
                        means[patch_i])
                    
                    # create a y^ that contains both the patch to match and the overlap
                    new_patch = np.concatenate((patch, prev_overlap))
                    
                    # mask Dh
                    masks = np.zeros(Dh.shape)
                    masks[:] = mask
                    masked_Dh = masks*Dh
                    
                    # create a D^
                    D_carrot = np.concatenate((Dl, masked_Dh),axis=1)
                    
                    #fit
                    D_carrot_t = np.transpose(D_carrot)
                    trainer.fit(D_carrot_t, new_patch)
                    
                    # reconstruct image
                    reconstructed_patch = Dht.dot(trainer.coef_.reshape(atoms,1)).reshape(highPatchSize) + means[patch_i]
                    reconstructed_image[i:i+highPatchSize[0], j:j+highPatchSize[1]] += reconstructed_patch
                    reconstructed_multiples[i:i+highPatchSize[0], j:j+highPatchSize[1]] += 1
                    
            reconstructed_image /= reconstructed_multiples
            reconstructed_image*=255
            reconstructed_image[reconstructed_image<0] = 0
            reconstructed_image[reconstructed_image>255] = 255
            reconstructed_image = reconstructed_image.astype(np.uint8)
            reconstruction[:,:,channel] = reconstructed_image
            
        dirs = image_path.split(os.path.sep)
        print(dirs)
        fileName = dirs[-1]
        print(fileName)
        
        img_tl.save_as_image(reconstruction, "2x_" + fileName)


def GenOverlapMask(patch, patchCount, mean):
    # generate overlap and mask for Dh
    prev_overlap = np.copy(patch)
    # remove the added means
    prev_overlap = prev_overlap - mean*patchCount
    # get the average patch result
    prev_overlap = prev_overlap/patchCount
    # deal with nan values
    prev_overlap[np.isnan(prev_overlap)] = 0
    
    mask = np.copy(prev_overlap)
    mask[mask!=0] = 1
    
    prev_overlap=prev_overlap.reshape(-1)
    mask = mask.reshape(-1)
    
    return prev_overlap, mask

    