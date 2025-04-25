import numpy as np
import torch
import torch.nn as nn
    
from utils import dice_score, dice_loss
    
def evaluate(model, data, device):
    val_losses = []
    val_dcs = []
    criterion = nn.BCELoss()
    with torch.no_grad():
        model.eval()
        for batch in data:
            image = batch["image"].to(device)
            mask = batch["mask"].to(device)
            pred_mask = model(image)
            loss = criterion(pred_mask, mask).item()
            dc_loss = dice_loss(pred_mask, mask).item()
            dc = dice_score(pred_mask, mask).item()
            val_losses.append(loss+dc_loss)
            val_dcs.append(dc)
        print(f"val losses: {np.mean(val_losses)}, val dice score: {np.mean(val_dcs)}")
    return val_losses, val_dcs
        