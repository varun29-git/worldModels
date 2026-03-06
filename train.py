import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import pickle
from PIL import Image

from model import VAE


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class CarRacingDataset(Dataset):

    def __init__(self, dataset_path):
        
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

        images = []

        for episode in data:
            images.extend(episode["observations"])

        self.images = images

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64,64)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx]

        img = self.transform(img)

        return img


# --------------------------------------------------
# Loss
# --------------------------------------------------

def vae_loss(recon_x, x, mu, logvar):

    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl


# --------------------------------------------------
# Training
# --------------------------------------------------

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CarRacingDataset("carRacing_dataset.pkl")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = VAE(input_shape=(3,64,64), latent_dim=32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 20

    for epoch in range(epochs):

        total_loss = 0

        for batch in dataloader:

            batch = batch.to(device)

            recon, mu, logvar = model(batch)

            loss = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(dataset)}")


    torch.save(model.state_dict(), "vae.pt")


if __name__ == "__main__":
    train()