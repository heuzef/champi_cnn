# Configuration d'une VM Ubuntu Serveur pour le projet Champi

vim /etc/hostname # Modification du nom d'hôte
apt upgrade && apt update # MAJ du système
apt install -y curl wget git tmux vim nano tree unzip nmap apache2 network-manager net-tools python3 python3-pip python3-virtualenv # Installation de quelques outils
passwd # Changement du mot de passe
adduser champi # Création de l'utilisateur "champi"
usermod -aG sudo champi # Ajout de l'utisateur champi au groupe admin
cat /etc/ssh/ssh_host_ed25519_key.pub | ssh -p23 -i /etc/ssh/ssh_host_ed25519_key u353969-sub1@u353969.your-storagebox.de install-ssh-key # Autorisation de la clef ssh auprès du serveur de stockage des data
ssh -p23 u353969-sub1@u353969.your-storagebox.de -i /etc/ssh/ssh_host_ed25519_key # Test de connexion au serveur de stockage

crontab -e # Ajout de la commande de montage automatique suivante dans le crontb au démarrage du système :
# @reboot sshfs -o IdentityFile=/etc/ssh/ssh_host_ed25519_key -o allow_other -p 23 u353969-sub1@u353969.your-storagebox.de:/home/ /data/
# Ce n'est pas la meilleur façon de procéder, mais cela à le mérite d'être simple
crontab -l # Verification

# Clone du dépôt
sudo su
cd /home/champi/jan24_cds_mushrooms/
eval "$(ssh-agent -s)"
ssh-add /etc/ssh/ssh_host_ed25519_key
git clone git@github.com:DataScientest-Studio/jan24_cds_mushrooms.git

# Pull du dépôt (en root)
eval "$(ssh-agent -s)" ; ssh-add /etc/ssh/ssh_host_ed25519_key ; cd /home/champi/jan24_cds_mushrooms/ ; git pull

# Initialisation de l'environnement python

# pip install virtualenv
# virtualenv venv
# source venv/bin/activate
pip install -r requirements.txt
pip list