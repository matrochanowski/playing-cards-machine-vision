import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from Classifier import Classifier


# Dataset definition
class ImageDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Read file names and determine labels
        class_0_paths = []
        class_1_paths = []

        for file_name in os.listdir(folder_path):
            if file_name.startswith('0'):
                class_0_paths.append(os.path.join(folder_path, file_name))
            elif file_name.startswith('1'):
                class_1_paths.append(os.path.join(folder_path, file_name))

        # Undersampling: Limit the number of class 0 samples to match class 1
        min_samples = min(len(class_0_paths), len(class_1_paths))
        class_0_paths = class_0_paths[:min_samples]
        class_1_paths = class_1_paths[:min_samples]

        self.image_paths = class_0_paths + class_1_paths
        self.labels = [0] * len(class_0_paths) + [1] * len(class_1_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, label


# Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])

# Load dataset
folder_path = 'cut_images'
dataset = ImageDataset(folder_path, transform=transform)

# Split into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model
print(torch.cuda.is_available())  # Powinno zwrócić True
print(torch.cuda.device_count())  # Powinno zwrócić liczbę dostępnych GPU
print(torch.cuda.get_device_name(0))  # Powinno wyświetlić nazwę karty graficznej
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Classifier().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        print(f"Is model on GPU? {next(model.parameters()).is_cuda}")  # Powinno zwrócić True

        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")


# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs=10)


# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


# Evaluate the model
evaluate_model(model, test_loader)
torch.save(model, os.path.join(os.pardir, "models", "binary_classifierV2.pth"))
