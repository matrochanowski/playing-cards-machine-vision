import torch
from usage_image_json import id_to_class
from torchvision import transforms
import cv2
from models.SimpleCNN import SimpleCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("models/model.pth")
model.to(device)
model.eval()

IMAGE_SIZE = (64, 64)

# Transformacje obrazów
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

image_path = "cropped/images/krol czerwo.png"
image = cv2.imread(image_path)
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

with torch.no_grad():  # wyłączenie gradienów, bo nie potrzebujemy ich do predykcji
    output = model(image)

# Uzyskaj klasę z najwyższym wynikiem
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()

predicted_class = id_to_class(predicted_class + 1)
print(f"Predicted class: {predicted_class}")
