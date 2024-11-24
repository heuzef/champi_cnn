# Import des librairies
from PIL import Image
import cv2
import os
import logging
from datetime import datetime
import torch

# Configuration de la journalisation pour suivre le processus et les erreurs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chargement du modèle YOLOv5 pré-entraîné depuis le hub de modèles de PyTorch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def process_images(base_path, html_path, out_path):
    # Je crée une liste pour stocker les entrées de données pour le fichier HTML
    html_data = []
    
    # Je parcours chaque dossier représentant une espèce dans le répertoire de base
    for species_folder in os.listdir(base_path):
        species_path = os.path.join(base_path, species_folder)
        dest_path = os.path.join(out_path, species_folder)
        if os.path.isdir(species_path):
            for image_file in os.listdir(species_path):
                try:
                    image_path = os.path.join(species_path, image_file)
                    # Je charge l'image à partir du disque
                    image = Image.open(image_path)
                    # Je convertis l'image en format attendu par le modèle
                    results = model(image)
                    # Je récupère les boîtes englobantes des prédictions du modèle
                    boxes = results.xyxy[0]  # Coordonnées des boîtes sous forme de Tensor
                    if len(boxes) > 0:
                        box = boxes[0]
                        # Je convertis les coordonnées des boîtes en entiers pour le recadrage
                        x1, y1, x2, y2 = int(box[0].item()), int(box[1].item()), int(box[2].item()), int(box[3].item())
                        # Je recadre l'image selon la boîte englobante
                        cropped_image = image.crop((x1, y1, x2, y2))
                        # Je redimensionne l'image pour l'analyse standard en vision par ordinateur
                        cropped_image = cropped_image.resize((224, 224))
                        # Je construis le nouveau chemin de l'image avec des informations détaillées
                        # new_image_path = f'{species_path}/{image_file[:-4]}_cropped_{x1}_{y1}_{x2}_{y2}.jpg'
                        new_image_path = f'{dest_path}/{image_file[:-4]}.jpg'
                        cropped_image.save(new_image_path)
                        logging.info(f'Image processed and saved: {new_image_path}')
                        # Je stocke les informations pour le fichier HTML
                        html_data.append((species_folder, image_file, new_image_path, x1, y1, x2, y2))
                    else:
                        logging.warning(f'No bounding box found for image: {image_path}')
                except Exception as e:
                    logging.error(f'Error processing image {image_path}: {e}')
    
    # Je génère un fichier HTML pour visualiser les résultats
    generate_html(html_data, html_path)

def generate_html(data, file_path):
    # Je crée un fichier HTML pour afficher les images et leurs informations
    with open(file_path, 'w') as file:
        file.write('<html><head><title>Image Processing Results</title></head><body>')
        file.write('<h1>Results of Mushroom Image Processing</h1>')
        file.write('<table border="1"><tr><th>Species</th><th>Original Image</th><th>Processed Image</th><th>Box Coordinates</th></tr>')
        for entry in data:
            species, original, processed, x1, y1, x2, y2 = entry
            file.write(f'<tr><td>{species}</td><td><img src="{original}" width="100"></td><td><img src="{processed}" width="100"></td><td>({x1}, {y1}, {x2}, {y2})</td></tr>')
        file.write('</table></body></html>')


# Créer les dossiers des espèces dans le dossier de destination
for species_folder in os.listdir(base_path):
    species_path = os.path.join(base_path, species_folder)
    dest_path = os.path.join(out_path, species_folder)
    os.makedirs(dest_path, exist_ok=True)

if __name__ == "__main__":
    # Je définis le chemin de base pour les images et le chemin pour sauvegarder le fichier HTML
    base_path = '/data/LAYER0/MO/MO/'
    out_path = '/data/LAYER1/MO/MO/'
    html_file_path = '/data/LAYER1/MO/boxing_yolo_v5.html'

    # Créer les dossiers des espèces dans le dossier de destination
    for species_folder in os.listdir(base_path):
        species_path = os.path.join(base_path, species_folder)
        dest_path = os.path.join(out_path, species_folder)
        os.makedirs(dest_path, exist_ok=True)

    # J'exécute la fonction de traitement des images
    process_images(base_path, html_file_path, out_path)