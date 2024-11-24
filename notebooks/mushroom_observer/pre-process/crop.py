from PIL import Image
import os



if not os.path.isdir("data/cropped"):
    os.makedirs("data/cropped")

species = os.listdir("data/original")

for specie in species:
    if not os.path.isdir("data/cropped/"+specie):
        os.makedirs("data/cropped/"+specie)
    fichiers = os.listdir("data/original/"+specie)
    for fichier in fichiers:
        if not os.path.isfile("data/cropped/"+specie+"/"+fichier):
            image = Image.open("data/original/"+specie+"/"+fichier)
            largeur, hauteur = image.size
            if(largeur > hauteur):
                margin = (largeur - hauteur) / 2
                if(specie == '382'):
                    margin = 0
                zone_recadrage = (margin, 0, hauteur+margin, hauteur)  
            else:
                margin = (hauteur - largeur) / 2
                if(specie == '382'):
                    margin = 0
                zone_recadrage = (0, margin, largeur, largeur+margin)  
            image_recadree = image.crop(zone_recadrage)
            image_recadree.save("data/cropped/"+specie+"/"+fichier)
