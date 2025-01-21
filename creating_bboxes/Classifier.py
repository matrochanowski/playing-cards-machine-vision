from torch import nn
from torchvision import models


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Use a pretrained ResNet18 model
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)  # Replace the classifier for binary output

    def forward(self, x):
        return self.model(x)
