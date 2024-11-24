from PIL import Image
import os



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
            
            image_resizee = image.resize((224, 224))
            image_resizee.save("data/vgg-16/"+specie+"/"+fichier)
