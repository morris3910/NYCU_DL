def dice_score(pred_mask, gt_mask, epsilon = 1e-6):
    # implement the Dice score here
    import torch

    pred_mask[pred_mask > 0.5] = torch.tensor(1.0)
    pred_mask[pred_mask <= 0.5] = torch.tensor(0.0)

    common_pix = (abs(pred_mask - gt_mask) < epsilon).sum()
    pred_img_pix = pred_mask.reshape(-1).shape[0]
    gt_img_pix = gt_mask.reshape(-1).shape[0]
    dice_score = 2 * common_pix / (pred_img_pix + gt_img_pix)
    
    return dice_score

def dice_loss(pred_mask, gt_mask, epsilon=1e-6):
    import torch
    intersection = torch.sum(gt_mask * pred_mask) + epsilon
    union = torch.sum(gt_mask) + torch.sum(pred_mask) + epsilon
    return 1 - (2 * intersection / union)

def plot_img(model2, data_path, model):
    from tqdm import tqdm
    from torchvision.utils import save_image
    import numpy as np
    import os
    from PIL import Image
    list_path = data_path + '/annotations/test.txt'
    with open(list_path) as f:
        filenames = f.read().strip('\n').split('\n')
    filenames = [x.split(' ')[0] for x in filenames]
    
    os.makedirs(f'{data_path}outputs_imgs/{model}', exist_ok=True)
    for i, file in tqdm(enumerate(filenames)):
        img_path = data_path + '/images/' + file + '.jpg'
        data = preprocess_data(img_path)
        data = data.unsqueeze(0).to("cuda")
        mask = model2(data).cpu().detach().numpy().reshape(256, 256)
        mask = mask > 0.5
        new_img = to_img(data, mask)
        new_img.save(f'{data_path}outputs_imgs/{model}/{i+1}_mask.png')
    

def preprocess_data(img_path):
    from PIL import Image
    import numpy as np
    import torch
    data = Image.open(img_path).convert("RGB")
    data = np.array(data.resize((256, 256), Image.BILINEAR))
    data = torch.tensor(data, dtype=torch.float32)
    data /= 255
    data = torch.permute(data, (2, 0, 1))
    return data
        
def to_img(data, mask):
    from PIL import Image
    import numpy as np
    data = data.squeeze(0).cpu().numpy().transpose((1, 2, 0))
    mask = np.stack((mask,)*3, axis=-1)
    data = data * 255
    mask = mask * 255
    mask = mask.astype('uint8')
    mask = Image.fromarray(mask)
    data = Image.fromarray(data.astype('uint8'))
    return Image.blend(data, mask, alpha=0.5)