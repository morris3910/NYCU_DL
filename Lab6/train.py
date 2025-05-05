import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from dataset import ICLEVRDataset
from model import ConditionalDDPM
import numpy as np
from tqdm import tqdm
import wandb

def save_checkpoint(model, optimizer, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path}")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-5
T = 1000 # time stamp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConditionalDDPM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

scheduler = DDPMScheduler(
    num_train_timesteps = T,
    beta_schedule = "squaredcos_cap_v2",
    prediction_type = "epsilon"
)

train_dataset = ICLEVRDataset('train.json', img_dir='../iclevr/iclevr')
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
wandb.init(project="DL_Lab6_DDPM", name="conditional_ddpm_run", mode = "offline")

# Training
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs = imgs.to(device)
        labels = labels.to(device)
        batch_size = imgs.size(0) # shape[0]

        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(imgs)
        noisy_imgs = scheduler.add_noise(imgs, noise, timesteps)

        pred_noise = model(noisy_imgs, timesteps, labels)
        loss = nn.MSELoss()(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        wandb.log({"batch_loss": loss.item()})

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

    wandb.log({"epoch_loss": avg_loss})

    # Save
    if epoch%1 == 0:
        save_checkpoint(model, optimizer, f"./results/ddpm_epoch{epoch+1}.pth")

# Finish 
wandb.finish()
