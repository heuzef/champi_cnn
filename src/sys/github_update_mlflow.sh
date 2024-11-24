#!/bin/bash

# Chemin vers les datas mlflow
mlflow_path="../../models/"

# Obtenir la date du jour au format AAAA-MM-JJ
date_today=$(date +%Y%m%d)

# Nom du nouveau dossier avec la date
new_dir="${mlflow_path}mlflow_${date_today}"

# Créer le nouveau dossier
mkdir -p "$new_dir"

# Nom de l'archive
backup_file=backup_mlflow_${date_today}.tar.gz

# Changer de répertoire
cd "$new_dir"

# Créer l'archive
tar -czf $backup_file "${mlflow_path}mlruns/" "${mlflow_path}mlartifacts/"

# Découper l'archive pour la limite de Github
split -b 20m $backup_file $backup_file.

# Supprime l'archive trop volumineuse pour Github
rm $backup_file

# Envoi sur Github
git add "$new_dir"
git commit -m "Backup MLFlow ${date_today}"
git push

# Ultérieurement, pour récupérer l'archive si besoin :
# cd "$new_dir" ; cat $backup_file.* > $backup_file