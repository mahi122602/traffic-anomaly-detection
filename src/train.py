import os
print("Training script started")

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from model import Autoencoder


# Limit dataset size for faster training
MAX_IMAGES = 10000


class FrameDataset(Dataset):

    def __init__(self, frame_folder):

        self.image_paths = []

        print("Scanning dataset...")

        for folder in os.listdir(frame_folder):

            folder_path = os.path.join(frame_folder, folder)

            if not os.path.isdir(folder_path):
                continue

            for img in os.listdir(folder_path):

                self.image_paths.append(os.path.join(folder_path, img))

                # Stop when max images reached
                if len(self.image_paths) >= MAX_IMAGES:
                    break

            if len(self.image_paths) >= MAX_IMAGES:
                break

        print("Total frames used for training:", len(self.image_paths))

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = self.transform(img)

        return img


dataset = FrameDataset("data/frames")

print("Dataset ready")

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


model = Autoencoder()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


epochs = 5

print("Starting training...")

for epoch in range(epochs):

    total_loss = 0

    for images in dataloader:

        outputs = model(images)

        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss:", total_loss)


os.makedirs("models", exist_ok=True)

torch.save(model.state_dict(), "models/anomaly_model.pth")

print("Model training complete")