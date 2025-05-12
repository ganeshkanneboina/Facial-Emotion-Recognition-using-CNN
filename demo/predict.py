import torch
import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from models.emotion_cnn import EmotionCNN
from torchvision import transforms

model = EmotionCNN()
model.load_state_dict(torch.load("outputs/best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

img = cv2.imread("demo/sample_face.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = transform(img).unsqueeze(0)
output = model(img_tensor)
pred = torch.argmax(output, 1)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("Predicted Emotion:", emotions[pred.item()])