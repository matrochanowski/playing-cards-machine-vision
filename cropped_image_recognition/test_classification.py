import numpy as np
import torch
from usage_image_json import id_to_class
from torchvision import transforms
import torch.nn.functional as F
import cv2
import os

# Ensure compatibility with GPU/CPU
print('loading model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = torch.load("../models/model-etiquette-2-03.pth")
model.to(device)
model.eval()
print('model loaded')


def prepare_the_picture(image):
    """
    Prepares an image for prediction by resizing, grayscaling, and converting to tensor format.
    """
    IMAGE_SIZE = (64, 64)

    # Transformations for the image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    return image


def classify_card(image: np.ndarray):
    """
    Classifies the card in the given image file path.
    Args:
        image (str): ndarray

    Returns:
        str: Predicted class of the card.
    """
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")

    image = prepare_the_picture(image)
    image = image.to(device)

    with torch.no_grad():  # Disable gradients for prediction
        output = model(image)

    probabilities = F.softmax(output, dim=1)

    # Get the class with the highest probability and its confidence
    confidence, predicted_class = torch.max(probabilities, 1)
    predicted_class = predicted_class.item()
    confidence = confidence.item()
    predicted_class_name = id_to_class(predicted_class)

    return predicted_class_name, confidence


if __name__ == "__main__":
    # Example usage when calling the file directly
    # image_name = 'AC.jpg'
    # image_path = os.path.abspath(os.path.join('..', 'cropped', 'images', image_name))
    image_path = '2S.png'
    print(f"Image path: {image_path}")
    image = cv2.imread(image_path)
    try:
        predicted_class = classify_card(image)
        print(f"Predicted class: {predicted_class}")
    except Exception as e:
        print(f"Error: {e}")
