import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from oxford_pet import load_dataset
from utils import dice_score, plot_img
from models.unet import UNet

def inference(args):
    model = torch.load(f"../saved_models/{args.model}")
    model.eval()
    model.to(args.device)
    data = load_dataset(args.data_path, mode="test")
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
    dice_scores = []
    progress = tqdm(enumerate(dataloader))
    for i, batch in progress:
        image = batch["image"].to(args.device)
        mask = batch["mask"].to(args.device)
        pred_mask = model(image)
        dice = dice_score(pred_mask, mask)
        dice_scores.append(dice.item())
        progress.set_description((f"iter: {i + 1}/{len(dataloader)}, Dice Score: {dice.item()}"))
    print(f"inference on {args.model}")
    print(f"Mean Dice Score: {np.mean(dice_scores)}")
    plot_img(model, args.data_path, args.model)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default="unet.pth", type=str)
    parser.add_argument('--device', default="cuda", type=str, help='device to use for training')
    parser.add_argument('--data_path', default="../dataset/oxford-iiit-pet/", type=str, help='path of the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)