import torch
from usage_image_json import id_to_class
from torchvision import transforms
import cv2
from models.BoundingBoxCNN import BoundingBoxCNN
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = torch.load("../models/model-object-detection.pth")
model.to(device)
model.eval()


def prepare_the_picture(image):
    IMAGE_SIZE = (256, 256)

    # Transformacje obrazów
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


image_name = 'test-od.jpg'
image_path = os.path.abspath(os.path.join('..', 'cropped', 'images', image_name))
print(image_path)
image = cv2.imread(image_path)
image = prepare_the_picture(image)
image = image.to(device)

with torch.no_grad():  # wyłączenie gradientów, bo nie potrzebujemy ich do predykcji
    output = model(image)

print(output)
