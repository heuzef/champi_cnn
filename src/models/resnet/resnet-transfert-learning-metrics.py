import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import mlflow
import mlflow.pytorch
import time
from tqdm import tqdm
from glob import glob
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Initialisation de l'URL
mlflow_server_uri = "http://192.168.1.19:8080"
mlflow.set_tracking_uri(mlflow_server_uri)
mlflow.set_experiment("champi_vgg16") # Le nom du projet

# Chemins vers les données
src_path_train = "data/LAYER2/MO/train"
src_path_test = "data/LAYER2/MO/validation"

NBCLASSES = len(os.listdir(src_path_train))
train_image_files = glob(src_path_train + '/*/*.jp*g')

# Transformations pour les images
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Fonction pour évaluer le modèle et calculer les métriques
def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Enregistrer les images mal classifiées
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    misclassified_images.append(inputs[i].cpu().numpy())
                    misclassified_labels.append(labels[i].cpu().numpy())
                    misclassified_preds.append(preds[i].cpu().numpy())

    # Calcul des métriques
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds)

    print(f"Accuracy: {accuracy}")
    print(f"F1-score: {f1}")
    print(f"Recall: {recall}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # Afficher les images mal classifiées
    for i in range(len(misclassified_images)):
        print(f"Image {i+1}: True label = {misclassified_labels[i]}, Predicted label = {misclassified_preds[i]}")

    return misclassified_images, misclassified_labels, misclassified_preds


# Fonction pour générer et afficher l'image de la matrice de confusion
def plot_confusion_matrix(conf_matrix, class_names):
    plt.figure(figsize=(14, 10))  # Augmenter la taille de la figure
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Ajouter des annotations
    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


for batch_size in [16]:

    # Charger les données
    train_dataset = ImageFolder(root=src_path_train, transform=transform_train)
    test_dataset = ImageFolder(root=src_path_test, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Charger le modèle pré-entraîné ResNet
    model = torchvision.models.resnet18(pretrained=True)

    # Remplacer la dernière couche pour correspondre au nombre de classes de votre jeu de données
    num_classes = len(train_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Déplacer le modèle sur le GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Fonction pour évaluer l'accuracy
    def evaluate_accuracy(model, data_loader, device):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total


    # Entraîner le modèle
    num_epochs = 100
    # Paramètres pour l'early stopping
    patience = num_epochs/10  # Nombre d'epochs sans amélioration avant d'arrêter l'entraînement
    best_accuracy = 0.0
    patience_counter = 0


    # Chemin du répertoire où le modèle sera sauvegardé
    model_dir = "saved_models/b0a8e6cd64a84fb39eac850094ddb8a1"
    model_path = os.path.join(model_dir, 'best_model.pth')
    # Charger le meilleur modèle
    model.load_state_dict(torch.load(model_path))

    # Enregistrer le modèle dans MLflow
    #mlflow.pytorch.log_model(model, "model")

    # Évaluer le modèle final sur les données de test
    evaluate_model(model, test_loader, device)

