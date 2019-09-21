from PIL import Image
import os
from numpy import array

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
    
def get_patch(image, coord_start, height, width):
    x,y = coord_start
    patch = []
    data = image.getdata(0)
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
