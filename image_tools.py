from PIL import Image
import os
import numpy as np

ImagePath = "images/"

def importImage(path):
    
    return Image.open(path)
    
def xyConversion(x, y, image):
    img_x, img_y = image.size
    if x >= img_x or y >= img_y:
        raise xyOutOfBoundsError(x,y,img_x,img_y)
    return x + y*img_x

def get_images(rootPath=ImagePath):
    images = os.listdir(rootPath)
    return [rootPath+path for path in images]

def resizeImage(image, scale):
    width,height = image.size
    image.resize((width*scale, height*scale))

def save_as_image(arr, filename):
    a = Image.fromarray(arr)
    a.save(filename)

def get_patches(np_image, patch_size, stride=1):
    """ expects a 2d array contianing data organized in a row, column order
        returns patches row first
    """
    
    if (stride < 1):
        raise IndexError()

    height, width = np_image.shape[:2]
    
    patch_height, patch_width = patch_size
    
    if patch_height > height or patch_width > width:
        raise Exception("Patch is bigger than image")
    
    patches_shape = merge_tuples((((height - patch_height)//stride + 1)* ((width - patch_width)//stride + 1)),
                                 patch_size,
                                 np_image.shape[2:])
    
    patches = np.zeros(patches_shape)
    
    try:
        for i in range(0, height - patch_height + 1, stride):
            for j in range(0, width - patch_width + 1, stride):
                #print("{},{}".format(i,j))
                patches[i//stride*((width - patch_width)//stride + 1) + j//stride] = np_image[i:i+patch_height, j:j+patch_width]
                
    except TypeError:
        raise TypeError("ERROR: One of your inputs is of the wrong type. Most likely this is the stride")
    except IndexError:
        print("{},{}".format(i,j))
        print(patches_shape)
        raise IndexError()
        
    return patches

    # should have it auto calculate stride
def merge_patches(patches, image_size, stride=1):
    image = np.zeros(image_size)
    
    multiples = np.zeros(image_size)
    multiples.astype(np.uint)
    
    height, width = image_size
    patch_height, patch_width = patches.shape[1:]
    
    for i in range(0, height-patch_height + 1, stride):
        for j in range(0, width-patch_width + 1, stride):
            image[i:i+patch_height, j:j+patch_width] += patches[i//stride*((width - patch_width)//stride + 1) + j//stride]
            multiples[i:i+patch_height, j:j+patch_width] += 1
    
    return image/multiples
    
    
    
def merge_tuples(*t):
    """ taken from https://stackoverflow.com/questions/14745199/how-to-merge-two-tuples-in-python/14745275#14745275"""
    return tuple(j for i in (t) for j in (i if isinstance(i, tuple) else (i,)))

def get_patch(image, coord_start, height, width):
    x,y = coord_start
    patch = []
    
    for i in range(height):
        index = xyConversion(x,y+i,image)
        tArray = []
        for j in range(width):
            tArray.append(data[index+j])
        patch.append(tArray)
        
    patch = array(patch)
    return patch

class xyOutOfBoundsError(Exception):
    def __init__(self, x, y, maxX, maxY):
        self.message("Coordinates of " + str(x) + "," + str(y) + " are out of bounds\n"
                     + "size of image is: " + str(maxX) + "," + str(maxY) + ".\n")
