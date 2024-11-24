# Import des librairies
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
from tensorflow.keras import layers
import os
import math

# Import data
root = './data'
db = 'MO'
img_folder_path = root + "/vgg-16/"
augmented_dir = root + "/LAYER2/" + db
train_size_percent = 80


def augmentData(
        source_path,
        dest_path,
        train_size_percent,
        target_train_size = 10000,
        max_class_size = 0,
        min_class_size = 50,
        ):

    #clean de l'arborescence avant de commencer
    if os.path.isdir(dest_path):
        shutil.rmtree(dest_path)

    species = os.listdir(source_path)

    # Création des dossiers à partir de la liste dans le DataFrame
    for specie in species:
        images_source = os.listdir(img_folder_path+"/"+str(specie))
        if len(images_source) < min_class_size:
            continue;
        for split_folder in ["train", "validation", "test"]:
            folder_path = os.path.join(dest_path, split_folder, specie) # Construire le chemin complet du dossier
            os.makedirs(folder_path, exist_ok=True) # Créer le dossier, existe_ok=True permet de ne pas lever d'erreur si le dossier existe déjà


    # Configuration des paramètres pour l'augmentation des images
    data_augmentation = tf.keras.Sequential([
        # layers.Resizing(224, 224),                    # Redimensionnement déjà effectuée lors de l'étape de boxing
        layers.Rescaling(1./255),                       # Mise à l'échelle
        layers.RandomFlip("horizontal"),   # Flip
        layers.RandomRotation(0.2),                     # Rotation
        layers.RandomZoom((-0.2, 0.2)),                 # Zoom
        layers.RandomTranslation(0.2, 0.2),             # Translation
        #layers.RandomBrightness(factor= [-.001, .001]), # Ajustement de la luminosité
        #layers.RandomContrast(factor= .4),                   # Ajustement du contraste
    ])

    # Data Augmentation
    for specie in species: # Pour chaque espèce de champi
        images_source = os.listdir(img_folder_path+"/"+str(specie))
        if len(images_source) < min_class_size:
            continue;
        
        #calcul nb images de l'espèce courrante à mettre dans le jeu de train
        if(max_class_size == 0):
            train_count = round(len(images_source) * train_size_percent / 100)
        else:
            train_count = round(max_class_size * train_size_percent / 100)
            
        train_count_total = 0
        source_pictures_count = 0
        
        for pic in images_source:

            if(train_count > 0):
                split_dir = "train"
                train_count_total += 1
                #calcul nb images augmentées à générer pour chaque image source
                augment_ratio = math.floor( (target_train_size - train_count_total) / train_count)
            else:
                if(max_class_size == 0 or source_pictures_count < max_class_size):
                    split_dir = "validation"
                else:
                    split_dir = "test"

            # Chemin complet de l'image champi
            mushroom_pic_path = str(img_folder_path)+"/"+str(specie)+"/"+str(pic) 

            # Décodage de l'image JPEG en tant qu'image tensor, channels=3 pour une image couleur (RGB)
            image = tf.image.decode_jpeg(tf.io.read_file(mushroom_pic_path), channels=3) 

            #copie de l'image d'origine dans le dataset
            shutil.copy(mushroom_pic_path, str(dest_path)+"/"+split_dir+"/"+str(specie)+"/"+str(pic))

            if(train_count > 0):
                for i in range(augment_ratio): # Nombre d'images aumgentée générée par photo
                    # Générer l'image augmentée
                    augmented_image = data_augmentation(image)

                    # Enregistrer l'image au format JPEG
                    augmented_image_conv = tf.image.convert_image_dtype(augmented_image, tf.uint8) # Convertion
                    augmented_image_enc = tf.image.encode_jpeg(augmented_image_conv) # Encodage JPEG
                    fname = str(pic)+"_"+str(i)+".jpg" # Nommage
                    with open(str(dest_path)+"/"+split_dir+"/"+str(specie)+"/"+str(fname), 'wb') as f: f.write(augmented_image_enc.numpy()) # Écriture
                    train_count_total += 1
                    if(train_count_total == target_train_size):
                        break

            
            train_count -= 1
            source_pictures_count += 1



augmentData(img_folder_path, augmented_dir, 80, 1)