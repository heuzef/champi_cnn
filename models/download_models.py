# Download models from MLFLOW

# Libs
import os
import shutil
import pandas as pd
import requests

# Download fonction
def telecharger_fichier(url, file_name):
    """Télécharge un fichier depuis une URL et l'enregistre localement.

    Args:
        url (str): L'URL du fichier à télécharger.
        file_name (str): Le nom du fichier à enregistrer localement.
    """

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Le fichier a été téléchargé avec succès : {file_name}")
    else:
        print(f"Erreur {response.status_code} lors du téléchargement {file_name}")

# Data folder
data = "artifacts"

try:
    # Suppression du dossier existant (si présent)
    shutil.rmtree(data)
    print(f"Dossier existant '{data}' supprimé.")
except FileNotFoundError:
    pass # Le dossier n'existait pas, on passe à la création

# Création du nouveau dossier
os.mkdir(data)
print(f"Nouveau dossier '{data}' créé.")

# Load Names and URL
models = pd.read_csv("models.csv")

# Download
for model in range(len(models)):
    url = models.iloc[model]['url']
    file = data+"/"+models.iloc[model]['name']
    telecharger_fichier(url, file)