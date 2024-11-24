import os
from PIL import Image
import shutil

def prepareDataset(trainSize, testSize ):

    if os.path.isdir("data/prepared-dataset"):
        shutil.rmtree("data/prepared-dataset")

    os.makedirs("data/prepared-dataset")
    os.makedirs("data/prepared-dataset/train")
    os.makedirs("data/prepared-dataset/test")

    species = os.listdir("data/cropped")

    for specie in species:
        fichiers = os.listdir("data/cropped/"+specie)
        if len(fichiers) < trainSize+testSize:
            continue

        if not os.path.isdir("data/prepared-dataset/train/"+specie):
            os.makedirs("data/prepared-dataset/train/"+specie)

        if not os.path.isdir("data/prepared-dataset/test/"+specie):
            os.makedirs("data/prepared-dataset/test/"+specie)
        trainCount = trainSize
        testCount = testSize

        for fichier in fichiers:
            image = Image.open("data/cropped/"+specie+"/"+fichier)
            image_resizee = image.resize((224, 224))
            if trainCount > 0:
                image_resizee.save("data/prepared-dataset/train/"+specie+"/"+fichier)
                trainCount -=1
                continue 
            if testCount > 0:
                image_resizee.save("data/prepared-dataset/test/"+specie+"/"+fichier)
                testCount -=1
                continue
            break