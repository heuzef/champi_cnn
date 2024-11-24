import requests
from PIL import Image
import os
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input

if not os.path.isdir("data/vgg-16"):
    os.makedirs("data/vgg-16")

species = os.listdir("data/cropped")

for specie in species:
    if not os.path.isdir("data/vgg-16/"+specie):
        os.makedirs("data/vgg-16/"+specie)
    fichiers = os.listdir("data/cropped/"+specie)
    for fichier in fichiers:
        if not os.path.isfile("data/vgg-16/"+specie+"/"+fichier):
            image = Image.open("data/cropped/"+specie+"/"+fichier)
            zone_recadrage = (0, 0, 224, 224)  
            image_recadree = image.crop(zone_recadrage)
            image_normalisee = preprocess_input(np.array(image_recadree))
            image_normalisee.save("data/vgg-16/"+specie+"/"+fichier)