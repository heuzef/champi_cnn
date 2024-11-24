import requests
from PIL import Image
import os
import json

if not os.path.isdir("data/original"):
    os.makedirs("data/original")

fichiers = os.listdir("notebooks/mushroom_observer/dataset-mushroom-observer/qualified-dataset")
for fichier in fichiers:
    if(not fichier.startswith('specie-id-')):
        continue;
    
    specieId = fichier[10:fichier.index('.json')]
    if not os.path.isdir("data/original/"+specieId):
        os.makedirs("data/original/"+specieId)
    
    with open("notebooks/mushroom_observer/dataset-mushroom-observer/qualified-dataset/"+fichier, "r") as f:
        data = json.loads(f.read())
        imagesIds = data["upSelection"] + data["downSelection"]

        for imageId in imagesIds:
            if not os.path.isfile("data/original/"+specieId+"/"+imageId+".jpg"):
                response = requests.get("https://images.mushroomobserver.org/960/"+imageId+".jpg")
                if response.status_code == 200:
                    # Contenu de l'image (binaire)
                    contenu_image = response.content

                    # Ouvrir le fichier en mode binaire pour l'écriture
                    with open("data/original/"+specieId+"/"+imageId+".jpg", "wb") as fichier:
                        # Enregistrer le contenu de l'image dans le fichier
                        fichier.write(contenu_image)

