import image_tools as img_tl
import numpy as np
import scipy as sp
from sklearn.neural_network import MLPRegressor
import sys
import os
from joblib import dump, load

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


lowPatchSize = (3,3)
highPatchSize = (6,6)

def train(image_paths):

    est = MLPRegressor(
        warm_start = True,
        activation = "tanh",
        early_stopping = True)
    
    for image_path in image_paths:
        # import image_path
        try:
            print(image_path)
            img = img_tl.importImage(image_path)
        except IOError:
            print("Error, an image could not be found in the images directory. Did you move it while the machine was training?")
            sys.exit()
            
        img_height, img_width = img.size
        img_low,img = img_tl.halfImageResolutionForTraining(img, lowPatchSize, highPatchSize)
        img = np.array(img)
        img_low = np.array(img_low)
        
        for channel in range(img.shape[-1]):
        
            high_data = img[:,:,channel]
            low_data = img_low[:,:,channel]
            high_data = img_tl.convertImageDataToPatches(high_data, highPatchSize, 2)
            low_data = img_tl.convertImageDataToPatches(low_data, lowPatchSize)
            
            high_data_size = high_data.shape[1]
            low_data_size = low_data.shape[1]
            
            est.fit(low_data, high_data)
        
    dump(est, "models/neural_network.pickle")
    
def super_size(image_paths):
    
    est = load("models/neural_network.pickle")
    
    for image_path in image_paths:
        
        # import image
        try:
            print(image_path)
            img = img_tl.importImage(image_path)
        except IOError:
            print("Error, an image could not be found in the images directory. Did you move it while the machine was training?")
            sys.exit()
        
        img = np.array(img)
        img_height, img_width, channels = img.shape
        
        result = np.zeros((img_height*2, img_width*2, channels)).astype(np.uint8)
        
        for channel in range(img.shape[-1]):
            print(channel)
            patches = img_tl.get_patches(img[:,:,channel]/255, lowPatchSize)
            patches = patches.reshape(patches.shape[0], -1)
            means = np.mean(patches, axis=1).reshape(patches.shape[0], 1)
            patches = patches - means
            
            predicted_patches = est.predict(patches)
        
            predicted_patches = predicted_patches + means
            predicted_image = img_tl.merge_patches(
            predicted_patches.reshape(predicted_patches.shape[0], highPatchSize[0], highPatchSize[1]), 
                (img_height*2, img_width*2), 
                2)
            predicted_image[predicted_image<=0] = 0
            predicted_image[predicted_image>=1] = 1
            predicted_image *=255
            predicted_image = predicted_image.astype(np.uint8)
            
            result[:,:,channel] = predicted_image
         
        dirs = image_path.split(os.path.sep)
        fileName = dirs[-1]
        
        img_tl.save_as_image(result, "2x_" + fileName)
        
    