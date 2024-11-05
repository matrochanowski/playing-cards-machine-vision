import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from usage_image_json import load_all_images
from models.SimpleCNN import SimpleCNN
import numpy as np

# Dane wejściowe
all_images = load_all_images(less=10000)
all_images = [(image, label - 1) for image, label in all_images]

# Ustalony rozmiar do przeskalowania obrazów
IMAGE_SIZE = (64, 64)

# Transformacje obrazów
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])


# Definicja zbioru danych
class CustomImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, label = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Stworzenie pełnego zbioru danych
dataset = CustomImageDataset(all_images, transform=transform)

# Podział danych: 80% trening, 20% test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Tworzenie loaderów
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Przygotowanie modelu, funkcji straty i optymalizatora
num_classes = len(set(label for _, label in all_images))
model = SimpleCNN(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Funkcja treningowa
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")


# Funkcja testująca
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():  # Wyłączanie gradientów w trybie testowym
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on the test data: {accuracy:.2f}%")


# Uruchomienie treningu i testowania
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Przenieś model na GPU
model.to(device)

# Trening modelu
train(model, train_loader, criterion, optimizer, epochs=30)

# Testowanie modelu
test(model, test_loader)

model_path = "models\\model.pth"
torch.save(model, model_path)
