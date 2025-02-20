import torch
import torch.nn as nn


class BoundingBoxCNN(nn.Module):
    def __init__(self):
        super(BoundingBoxCNN, self).__init__()

        # Warstwy konwolucyjne i pooling
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # (3, 256, 256) -> (32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (32, 256, 256) -> (32, 128, 128)
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (32, 128, 128) -> (64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (64, 128, 128) -> (64, 64, 64)
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # (64, 64, 64) -> (128, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (128, 64, 64) -> (128, 32, 32)
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # (128, 32, 32) -> (256, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256, 32, 32) -> (256, 16, 16)
            nn.BatchNorm2d(256)
        )

        # Warstwy gęste
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # Spłaszczenie (256, 16, 16) -> (256*16*16)
            nn.Linear(256 * 16 * 16, 512),  # Warstwa w pełni połączona
            nn.ReLU(),
            nn.Linear(512, 16)  # Wyjście: 4 bounding boxy po 4 wartości
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class SimpleBoundingBoxDetector(nn.Module):
    def __init__(self):
        super(SimpleBoundingBoxDetector, self).__init__()

        # Feature extractor (prosta sieć konwolucyjna)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Bounding box regressor (4 współrzędne na każdy region)
        self.box_regressor = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 4, kernel_size=1)  # 4 wartości: [xmin, ymin, xmax, ymax]
        )

    def forward(self, x):
        # Ekstrakcja cech
        features = self.feature_extractor(x)

        # Predykcja bounding boxów
        box_predictions = self.box_regressor(features)  # [batch, 4, H, W]

        # Reshape predykcji do listy propozycji
        batch_size, _, H, W = box_predictions.shape
        box_predictions = box_predictions.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)  # [batch, H*W, 4]

        return box_predictions
