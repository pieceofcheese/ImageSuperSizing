
import sys, getopt
import sparse
import neuralNet
import image_tools as img_tl


helpText = """SuperSize.py [options] image
    -h Prints help message
    -m Which Model to use
    -d Directory to look for images
    
    Models:
        sparse : Based on Jiachoa Yang et. al Sparse dictionary model
        neural : a neural network based model
"""

def main(argv):
    
    model = "neural"
    
    try:
        opts, args = getopt.getopt(argv,"hm:d:")
    except getopt.GetoptError:
        print("SuperSize.py [options] image")
        sys.exit(2)
    
    # handle options
    for opt, arg in opts:
        if opt == '-h':
            print (helpText)
            sys.exit()
        elif opt == "-m":
            model = arg
        elif opt == "-d":
            images += img_tl.get_images(arg)
    
    images = []
    # handle arguments
    if len(args) > 1:
        for arg in args:
            images.append(arg)
    elif(len(args)>0):
        images += args
    else:
        sys.exit()
    
    if model == "sparse":
        # do model
        sparse.super_size(images)
        
    elif model == "neural":
        neuralNet.super_size(images)
    else:
        # do a default model
        print("Error: you did not select an existing model")
        
if __name__== "__main__":
    main(sys.argv[1:])
