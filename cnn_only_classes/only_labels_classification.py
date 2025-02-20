from torch.utils.data import DataLoader, Dataset
import json
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import random_split


class OnlyLabelsImageDataset(Dataset):
    def __init__(self, folder_path, transform=None, how_many=20_000):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []

        json_filepath = os.path.join(os.pardir, 'jsons', 'images.json')
        with open(json_filepath, 'r') as jsonfile:
            data = json.load(jsonfile)
        for i, entry in enumerate(data.values()):
            image_relative_path = entry['path']
            image_path = os.path.join(os.pardir, 'train', 'images', image_relative_path)

            annotations = entry['annotations']
            labels_list = list()
            for annotation in annotations:
                labels_list.append(annotation['class_id'] - 1)
            label = 52 * [0]

            for value in labels_list:
                label[value] = 1

            self.image_paths.append(image_path)
            self.labels.append(label)

            if i > how_many - 1:
                break

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Convert label to PyTorch tensor
        label = torch.tensor(label, dtype=torch.float32)

        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)

        return image, label


class CNNMultiLabel(nn.Module):
    def __init__(self, num_classes=52):
        super(CNNMultiLabel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 512),  # Adjust based on input image size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
            nn.Sigmoid(),  # For multi-label classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def calculate_quality_metrics(model, dataloader):
    """Calculates precision, recall, F1-score for the model."""
    model.eval()  # Set model to evaluation mode
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            predictions = (outputs > 0.05).float()  # Convert probabilities to binary predictions

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Calculate metrics
    all_labels = torch.tensor(all_labels).numpy()
    all_predictions = torch.tensor(all_predictions).numpy()

    precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)

    return precision, recall, f1


# Training script
def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.float32)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print epoch loss
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

        # Calculate quality metrics on validation data
        precision, recall, f1 = calculate_quality_metrics(model, val_loader)
        print(f"Quality Scores - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    print("Training Complete")


batch_size = 32
learning_rate = 0.001
num_epochs = 30
num_classes = 52

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = OnlyLabelsImageDataset(folder_path="path_to_folder", transform=transform, how_many=12_000)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
model = CNNMultiLabel(num_classes=num_classes).to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_size = int(0.95 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Train the model with metrics
train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)
torch.save(model, "../models/all_labels_model.pth")
