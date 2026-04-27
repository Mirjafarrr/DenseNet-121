import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

from src.model import DenseNet

def predict_image(image_url):
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

    model = DenseNet().to(device)
    model.load_state_dict(torch.load("models/best_densenet.pth", map_location=device))
    model.eval() 

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(device) 

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        probabilities = torch.nn.functional.softmax(output, dim=1)

    class_idx = predicted.item()
    confidence = probabilities[0][class_idx].item() * 100
    
    print(f"\n--- Prediction Result ---")
    print(f"Predicted Class: {classes[class_idx].upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"--------------------------\n")

if __name__ == "__main__":
    test_url = "https://www.startpage.com/av/proxy-image?piurl=https%3A%2F%2Fimage-tc.galaxy.tf%2Fwijpeg-6keue73dqvwpbaabt957aqg6l%2Fuk-deer-species-red-deer_standard.jpg%3Fcrop%3D57%252C0%252C867%252C650%26width%3D928&sp=1777280466T1a6141d06304715a6073ecba8c78ad51b821e7e05a8d769f2831714d09ccc8ea"
    predict_image(test_url)