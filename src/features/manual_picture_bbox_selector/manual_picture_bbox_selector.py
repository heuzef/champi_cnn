############################
# Auteur :  Heuzef (heuzef.com)
# Date : 05/2024
# Description : MPBS (Manual Picture BBOX Selector)
#
# Import des librairies ####
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
#
# Fonctions ################
#
def drawRectangle(frame, bbox):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
#
def displayRectangle(frame, bbox):
    plt.figure(figsize=(20, 10))
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    plt.imshow(frameCopy)
    plt.axis("off")
#
def saveBoxed(frame, bbox, path):
    frameCopy = frame.copy()
    drawRectangle(frameCopy, bbox)
    frameCopy = cv2.cvtColor(frameCopy, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, frameCopy)
    # plt.imshow(frameCopy) # Preview
############################


# Repertoires des donnés
pic_path = './pictures/'
boxed_path = './boxed/'

# Liste des photos
pic_files = os.listdir(pic_path)

# Explorer et créer une liste des fichiers
pic_name = []
for name in pic_files:
    pic_name.append(name)

# Création du dataframe
data = {'picture': pic_name, 'x1': 0, 'x2': 0, 'y1': 0, 'y2': 0}
pictures_bbox =  pd.DataFrame(data)

# Traitement des images présente dans le dossier
for pic in pictures_bbox.index:

    # Selection manuel sur l'image
    frame = cv2.imread(pic_path+pictures_bbox['picture'][pic])
    bbox = cv2.selectROI(pictures_bbox['picture'][pic], frame, showCrosshair=False, fromCenter=False)

    # Sauvegarde de l'i
    # mage selectionnée
    saveBoxed(frame=frame, bbox=bbox, path=boxed_path+pictures_bbox['picture'][pic])

    # Sauvegarde des points de coordonnées
    pictures_bbox.loc[pic, 'x1'] = bbox[0]
    pictures_bbox.loc[pic, 'x2'] = bbox[1]
    pictures_bbox.loc[pic, 'y1'] = bbox[2]
    pictures_bbox.loc[pic, 'y2'] = bbox[3]

    cv2.destroyAllWindows()

# Exportation du DataFrame au format CSV
pictures_bbox.to_csv('pictures_bbox.csv', index=False)
print("Fichier CSV généré ! ")