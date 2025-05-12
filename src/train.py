import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.emotion_cnn import EmotionCNN
from data.dataloader import FERDataset
from utils.metrics import evaluate_model, plot_confusion
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

train_dir = "data/archive/train"
val_dir = "data/archive/test"  # Assuming test folder is used as val

transform = transforms.Compose([
    transforms.Grayscale(),  # Ensures single channel
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_data = ImageFolder(train_dir, transform=transform)
val_data = ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)

emotion_labels = train_data.classes
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "outputs/best_model.pth")