import argparse
import torch.nn as nn
from tqdm import tqdm
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from oxford_pet import load_dataset
from evaluate import evaluate
from models.unet import UNet
from models.resnet34_unet import resnet_34_unet
from utils import dice_score, dice_loss

def train(args):
    # implement the training function here
    train_data = load_dataset(args.data_path, mode="train")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data = load_dataset(args.data_path, mode="valid")
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    if args.model == "unet":
        model = UNet(in_channels=3, num_classes=1).to(args.device)
    else:
        model = resnet_34_unet(in_channels=3, num_classes=1).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    writer = SummaryWriter(f"runs/{args.model}/")
    torch.autograd.set_detect_anomaly(True)
    best_dice_score = 0.8
    for epoch in range(args.epochs):
        train_losses = []
        train_dcs = []
        model.train()
        progress = tqdm(enumerate(train_loader))
        for idx, batch in progress:
            img = batch["image"].to(args.device)
            mask = batch["mask"].to(args.device)
            y_pred = model(img)
            loss = criterion(y_pred, mask) + dice_loss(y_pred, mask)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            dc = dice_score(y_pred, mask)
            train_dcs.append(dc.item())
            progress.set_description((f"Epoch: {epoch + 1}/{args.epochs}, iter: {idx + 1}/{len(train_loader)}, Loss: {np.mean(train_losses):.4f}, Dice Score: {np.mean(train_dcs):.4f}"))

        val_losses, val_dcs = evaluate(model, val_loader, args.device)
        scheduler.step()
        writer.add_scalars(f"Loss", {"train": np.mean(train_losses), "valid": np.mean(val_losses)}, epoch)
        writer.add_scalars(f"Dice Score", {"train": np.mean(train_dcs), "valid": np.mean(val_dcs)}, epoch)
        if np.mean(val_dcs) > best_dice_score:
            best_dice_score = np.mean(val_dcs)
            torch.save(model, f"../saved_models/{args.model}.pth")
    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', default="unet", type=str, choices=["unet" ,"resnet34_unet"])
    parser.add_argument('--device', default="cuda", type=str, help='device to use for training')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)