import image_tools as img_tl
import numpy as np
from sklearn import linear_model
import sys
import math

lowPatchSize = (3,3)
highPatchSize = (6,6)
representationSize = (3,3)
atoms = 512
lmbda = 0.1

def train(image_paths):
    
    # make a Dh and Dl
    Dl = np.random.rand(lowPatchSize[0]*lowPatchSize[1], atoms)
    Dh = np.random.rand(highPatchSize[0]*highPatchSize[1], atoms)
        
    for image_path in image_paths:
        img = None
        try:
            img = img_tl.importImage(image_path)
        except IOError:
            print("Error, an image could not be found in the images directory. Did you move it while the machine was training?")
            sys.exit()
        img_x, img_y = img.size
        # ceil cause worried about 0 size
        img_low = img.resize((math.ceil(img_x*0.5), math.ceil(img_y*0.5)))
        
        # dictionary learning test
        dictionary_learning_test(
            Dl, 
            np.zeros((lowPatchSize[0]*lowPatchSize[1],1)), 
            img_tl.get_patch(img_low, (64,64), lowPatchSize[0], lowPatchSize[1]))
        
    
def dictionary_learning_test(D, Z, X):
    X = np.reshape(X, (lowPatchSize[0]*lowPatchSize[1], 1))
    Xt = np.transpose(X)
    Dt = np.transpose(D)
    Zt = np.transpose(Z)
    print(D)
    print(Z)
    print(X)
    print(Xt*X)
    print(Xt*D*Z)
    print(Zt*Dt*D*Z)
    print(Xt*X-2*Dt*Zt*X+Zt*Dt*D*Z)
    
    
        
    
def dictionary_learning(Dh, Dl, Z, N, M):
    
    # while not converged, loop
    
    # fix D, update Z
    # fix Z, update D
    pass

def super_size(Dh, Dl, Y):
    """Dh: high resolution dictionary
       Dl: low resolution dictionary
       Y:  low resolution image
    """
    
    maxX, maxY = Y.size
    patchX, patchY = patchSize
    
    # for each patch
    for i in range(maxY-patchY-1):
        for j in range(maxX-patchX-1):
            patch = img_tl.get_patch(Y, (i,j), patchY, patchX)
            
            meanPixelValue = np.mean(patch)
            
            
            
