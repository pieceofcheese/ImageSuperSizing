from PIL import Image
from image_tools import save_as_image
from image_tools import get_images
import sys, getopt
import numpy as np

root = "TestImages/"

def main(argv):

    try:
        opts, args = getopt.getopt(argv,"d:")
    except getopt.GetoptError:
        print("bwConvertor.py images...")

    images = []

    for opt, arg in opts:
        if opt == "-d":
            images += get_images(arg)

    if len(args) > 1:
        for arg in args:
            images += arg

    elif(len(args)>0):
        images += args

    print(images)

    for image in images:
        
        path = image.split("\\")
        filename = path[-1].split(".")
        name = filename[0]

        print(name)

        img = Image.open(image)

        img = np.array(img)
        save_as_image(img[:,:,0], root+name+"_red.png")
        save_as_image(img[:,:,1], root+name+"_green.png")
        save_as_image(img[:,:,2], root+name+"_blue.png")
            
if __name__== "__main__":
    main(sys.argv[1:])
