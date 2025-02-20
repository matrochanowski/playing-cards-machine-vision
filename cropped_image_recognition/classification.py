import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from torchvision import transforms
from usage_image_json import load_all_cropped_images, get_cropped_images, get_uneven_cropped_images
from models.SimpleCNN import SimpleCNN, EnhancedCNN
import os
import json
import cv2
import numpy as np
import random
from multiprocessing import freeze_support  # Dodaj tę linię

# Ustalony rozmiar do przeskalowania obrazów
IMAGE_SIZE = (64, 64)

# Transformacje obrazów
transform = transforms.Compose([
    transforms.ToTensor(),  # Bezpośrednio konwertuje NumPy array do tensora
])


# Definicja zbioru danych
class CustomImageDataset(Dataset):
    def __init__(self, dataset_dir, json_path, transform=None, less=25000):
        """
        Args:
            dataset_dir (str): Ścieżka do folderu z obrazami.
            json_path (str): Ścieżka do pliku JSON zawierającego informacje o obrazach i etykietach.
            transform (callable, optional): Transformacje do zastosowania na obrazach.
            less (int, optional): Maksymalna liczba obrazów do załadowania. Domyślnie 21200.
        """
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.less = less

        # Wczytaj dane z pliku JSON
        with open(json_path, 'r') as file:
            self.dict_data = json.load(file)

        # Przygotuj listę ścieżek do obrazów, bounding boxów i etykiet
        self.image_info = []
        for i in range(self.less):
            if str(i) in self.dict_data:
                file_path = os.path.join(self.dataset_dir, 'images', self.dict_data[str(i)]['path'])
                bboxes = [content['bbox'] for content in self.dict_data[str(i)]['annotations']]
                cats = [content['class_id'] for content in self.dict_data[str(i)]['annotations']]
                self.image_info.append((file_path, bboxes, cats))

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        # Pobierz informacje o obrazie
        file_path, bboxes, cats = self.image_info[idx]

        # Wczytaj i wytnij obrazki za pomocą funkcji get_cropped_images
        cropped_images = get_uneven_cropped_images(file_path, bboxes, cats)

        # Jeśli nie udało się wczytać obrazu, zwróć None
        if cropped_images is None:
            return None

        # Wybierz pierwszy wycięty obrazek i etykietę
        img, label = cropped_images[0]

        # Zmień rozmiar obrazka na 64x64 za pomocą OpenCV
        try:
            img = cv2.resize(img, (64, 64))
        except cv2.error:
            img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

            label = 0

        if self.transform:
            img = self.transform(img)

        return img, label


# Funkcja treningowa
def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    print('Training started.')
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
    return accuracy


if __name__ == '__main__':
    freeze_support()  # Dodaj tę linię

    # Stworzenie pełnego zbioru danych
    dataset_dir = '../train'
    json_dir = '../jsons/images.json'
    dataset = CustomImageDataset(dataset_dir, json_dir, transform=transform)

    # Podział danych: 80% trening, 20% test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Tworzenie loaderów
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    # Przygotowanie modelu, funkcji straty i optymalizatora
    num_classes = 53
    model = EnhancedCNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.007)

    # Uruchomienie treningu i testowania
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Przenieś model na GPU
    model.to(device)

    # Trening modelu
    train(model, train_loader, criterion, optimizer, epochs=400)

    # Testowanie modelu
    acc = test(model, test_loader)

    model_path = "../models/model-etiquette-2-03.pth"
    torch.save(model, model_path)

    text = f'Dokładność modelu {model_path} na danych testowych to {acc:.2f}%'
    with open('wynik_uczenia.txt', 'w') as outfile:
        outfile.write(text)

    os.system("shutdown /s /t 0")
