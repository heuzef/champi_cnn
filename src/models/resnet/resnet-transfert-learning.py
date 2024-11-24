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

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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


    # Démarrer un run MLflow
    with mlflow.start_run() as run:
        # Enregistrer les paramètres
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", 0.001)
        mlflow.log_param("momentum", 0.9)
        mlflow.log_param("patience", patience)
        mlflow.log_param("optimizer", "SDG")
        mlflow.log_param("nb_classes", NBCLASSES)
        mlflow.log_param("nb_images_train", len(train_image_files))

        # Chemin du répertoire où le modèle sera sauvegardé
        run_id = mlflow.active_run().info.run_id
        model_dir = f'saved_models/{run_id}'
        model_path = os.path.join(model_dir, 'best_model.pth')

        # Créer le répertoire parent s'il n'existe pas
        os.makedirs(model_dir, exist_ok=True)


        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            epoch_start_time = time.time()

            # Utiliser tqdm pour l'indicateur de progression
            with tqdm(train_loader, unit="batch") as tepoch:
                for inputs, labels in tepoch:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()

                    # Mettre à jour la barre de progression
                    tepoch.set_postfix(loss=running_loss / len(train_loader))

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            # Évaluer l'accuracy sur les données d'entraînement et de test
            train_accuracy = evaluate_accuracy(model, train_loader, device)
            test_accuracy = evaluate_accuracy(model, test_loader, device)

            # Enregistrer les métriques dans MLflow
            mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
            mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%, Time: {epoch_time:.2f}s')

            # Vérifier si l'accuracy sur le jeu de test s'est améliorée
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                patience_counter = 0
                # Sauvegarder le modèle
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1

            # Arrêter l'entraînement si l'accuracy ne s'améliore plus pendant plusieurs epochs
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # Charger le meilleur modèle
        model.load_state_dict(torch.load(model_path))

        # Enregistrer le modèle dans MLflow
        #mlflow.pytorch.log_model(model, "model")

        # Évaluer le modèle final sur les données de test
        final_test_accuracy = evaluate_accuracy(model, test_loader, device)
        print(f'Final Test Accuracy: {final_test_accuracy:.2f}%')
        evaluate_model(model, test_loader, device)

        # Enregistrer la précision finale dans MLflow
        mlflow.log_metric("best_val_accuracy", final_test_accuracy)

