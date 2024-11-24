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

# Initialisation de l'URL
mlflow_server_uri = "https://champi.heuzef.com"
mlflow.set_tracking_uri(mlflow_server_uri)
mlflow.set_experiment("champi_vgg16") # Le nom du projet

# Chemins vers les données
src_path_train = "data/LAYER2/MO/train"
src_path_test = "data/LAYER2/MO/validation"

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


# Charger les données
train_dataset = ImageFolder(root=src_path_train, transform=transform_train)
test_dataset = ImageFolder(root=src_path_test, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

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

# Paramètres pour l'early stopping
patience = 5  # Nombre d'epochs sans amélioration avant d'arrêter l'entraînement
best_accuracy = 0.0
patience_counter = 0


# Démarrer un run MLflow
with mlflow.start_run() as run:
    # Enregistrer les paramètres
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("momentum", 0.9)
    mlflow.log_param("patience", patience)

    # Entraîner le modèle
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Évaluer l'accuracy sur les données d'entraînement et de test
        train_accuracy = evaluate_accuracy(model, train_loader, device)
        test_accuracy = evaluate_accuracy(model, test_loader, device)

         # Enregistrer les métriques dans MLflow
        mlflow.log_metric("train_loss", running_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        
        # Vérifier si l'accuracy sur le jeu de test s'est améliorée
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            # Sauvegarder le meilleur modèle
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        # Arrêter l'entraînement si l'accuracy ne s'améliore plus pendant plusieurs epochs
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    
    # Charger le meilleur modèle
    model.load_state_dict(torch.load('best_model.pth'))

    # Enregistrer le modèle dans MLflow
    mlflow.pytorch.log_model(model, "model")

    # Évaluer le modèle final sur les données de test
    final_test_accuracy = evaluate_accuracy(model, test_loader, device)
    print(f'Final Test Accuracy: {final_test_accuracy:.2f}%')

    # Enregistrer la précision finale dans MLflow
    mlflow.log_metric("final_test_accuracy", final_test_accuracy)