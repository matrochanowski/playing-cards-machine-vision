import torch
from torchvision import transforms
from PIL import Image
import os
from Classifier import Classifier
import torch.nn.functional as F
import cv2

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to 128x128
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
])


# Function to prepare the image for classification
def prepare_image(image):
    image = image.convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image


# Load the trained model
print('loading model...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(os.path.join(os.pardir, "models", "binary_classifierV2.pth"))
model.to(device)
model.eval()
print('model loaded')

class_names = ['No card', 'There is a card']


# Function to classify an image and output the result
def classify_image(image: Image):
    # Prepare the image
    image = prepare_image(image)
    image = image.to(device)

    with torch.no_grad():
        # Perform forward pass
        outputs = model(image)

        probabilities = F.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(probabilities, 1)
        confidence = confidence.item()  # Convert to Python scalar

        # Get the class with the highest score
        _, predicted_class = torch.max(outputs, 1)
        predicted_class = predicted_class.item()

        # Print the predicted class
        if __name__ == "__main__":
            print(f"Predicted class: {class_names[predicted_class]}")
            print(f"Confidence: {confidence * 100:.2f}%")

        return predicted_class, confidence


# Main loop to input image paths and classify them
def main():
    while True:
        # Prompt the user for an image path
        image_path = input("Enter image path (or type 'exit' to quit): ")

        if image_path.lower() == 'exit':
            print("Exiting the program.")
            break

        if not os.path.exists(image_path):
            print("Invalid path. Please try again.")
            continue

        # Display the image using OpenCV (optional)
        image = cv2.imread(image_path)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Classify the image
        image = Image.fromarray(image).convert('RGB')
        classify_image(image)


# Run the main loop
if __name__ == "__main__":
    main()
