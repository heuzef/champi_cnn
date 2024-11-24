# Déploiement de MLflow pour le projet Champi

```shell
pip3 install mlflow==2.14.1
```

# Désactiver Apache (conflit de port avec ce service inutilisé)

```shell
systemctl stop apache2
systemctl disable apache2
```

# Copier les fichiers de configuration :

```shell
cp /home/champi/jan24_cds_mushrooms/src/sys/mlflow-server.service /etc/systemd/system/
cp /home/champi/jan24_cds_mushrooms/src/sys/mlflow-server.sh /data/
chmod +x /data/mlflow-server.sh 
```

# Activez et démarrez le service :

```shell
systemctl enable mlflow-server
systemctl start mlflow-server
systemctl status mlflow-server
```