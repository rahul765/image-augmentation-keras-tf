# -*- coding: utf-8 -*-
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from argparse import ArgumentParser
import numpy as np

# construct the argument parse and parse the argument
ap = ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image...")
ap.add_argument("-o", "--output", required=True, help="path to output directory...")
ap.add_argument("-p", "--prefix", type=str , default="image", help="output filename prefix...")
args = vars(ap.parse_args())

# Load the image
print("[INFO] Loading the image...")
image = load_img(args["image"])
#print("load_img",image)
image = img_to_array(image)
#print("img_to_array",image)
image = np.expand_dims(image, axis=0)
#print("expand_dims",image)


# construct the image generator for data augmentation then
# initialize the total number of images generated thus far
aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
                         height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")
total = 0

# construct the actual Python generator
print("[INFO] Generating images...")
imageGen = aug.flow(image, batch_size=1,save_to_dir=args["output"], save_prefix=args['prefix'],
                    save_format="png")

# Looping over examples from our data augmentation generator
for img in imageGen:
    # increment our counter
    total += 1
    
    # if we have reached 3 examples, break from the loop
    if(total==3):
        break