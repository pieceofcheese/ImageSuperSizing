
import sys, getopt
import sparse
import image_tools as img_tl


helpText = """train.py [options] image_folder
    -h Prints help message
    -m Which Model to train
"""

def main(argv):
    
    model = "sparse"
    
    try:
        opts, args = getopt.getopt(argv,"hm:")
    except getopt.GetoptError:
        print("train.py [options] image_folder")
        sys.exit(2)
    
    # handle options
    for opt, arg in opts:
        if opt == '-h':
            print (helpText)
            sys.exit()
        elif opt == "-m":
            model = arg
    
    images = []
    # handle arguments
    if len(args) > 0:
        for arg in args:
            images += img_tl.get_images(arg)
    else:
        images = img_tl.get_images()
    
    if model == "sparse":
        # do model
        sparse.train(images)
        
    else:
        # do a default model
        print("Error: you did not select an existing model")
        
if __name__== "__main__":
    main(sys.argv[1:])
