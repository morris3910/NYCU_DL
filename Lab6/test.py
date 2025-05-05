import os
import json
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from model import ConditionalDDPM
from dataset import ICLEVRDataset
from evaluator import evaluation_model
from diffusers import DDPMScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_ckpt_path = "./results/ddpm_sigmoid_epoch10.pth"     
save_dir = "./results_img"
os.makedirs(save_dir, exist_ok=True)
batch_size = 32
timesteps = 1000
guidance_weight = 1.0  

model = ConditionalDDPM().to(device)
model.load_state_dict(torch.load(model_ckpt_path, map_location=device, weights_only=True)['model_state_dict'])
model.eval()

scheduler = DDPMScheduler(
    num_train_timesteps=timesteps,
    beta_schedule="sigmoid",
    prediction_type="epsilon"
)
scheduler.set_timesteps(timesteps)

evaluator = evaluation_model()

test_dataset = ICLEVRDataset(json_path = "test.json", img_dir=None)
new_test_dataset = ICLEVRDataset(json_path = "new_test.json", img_dir=None)

test_labels = torch.stack([test_dataset[i] for i in range(len(test_dataset))])
new_test_labels = torch.stack([new_test_dataset[i] for i in range(len(new_test_dataset))])

def denormalize(x):
    return (x + 1) / 2

def sample_images(label_tensor, desc="Sampling"):
    model.eval()
    batch_size = label_tensor.size(0)
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    labels = label_tensor.to(device)
    for t in tqdm(scheduler.timesteps, desc=desc):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(x, t_tensor, labels)
        x = scheduler.step(noise_pred, t, x).prev_sample
    return x

def save_images(imgs, folder, prefix):
    imgs = denormalize(imgs)
    grid = make_grid(imgs, nrow=8)
    save_image(grid, os.path.join(folder, f"{prefix}_grid.png"))
    for idx, img in enumerate(imgs):
        save_image(img, os.path.join(folder, f"{prefix}_{idx:03d}.png"))

def evaluate_accuracy(imgs, labels):
    acc = evaluator.eval(imgs, labels)
    return acc

gen_test_imgs = sample_images(test_labels, desc="Sampling test set")
save_images(gen_test_imgs, save_dir, "test")

gen_new_imgs = sample_images(new_test_labels, desc="Sampling new_test set")
save_images(gen_new_imgs, save_dir, "new_test")

print("Evaluating...")
acc_test = evaluate_accuracy(gen_test_imgs, test_labels)
acc_new_test = evaluate_accuracy(gen_new_imgs, new_test_labels)
print(f"Accuracy on test.json: {acc_test:.2f}")
print(f"Accuracy on new_test.json: {acc_new_test:.2f}")

print("Generating denoising process...")

# label: ["red sphere", "cyan cube", "cyan cylinder"]
labels = [6, 9, 22]
label_vec = torch.zeros(24)
label_vec[[idx for idx in labels]] = 1
label_vec = label_vec.unsqueeze(0).to(device)

frames = []
x = torch.randn(1, 3, 64, 64, device=device)

with torch.no_grad():
    for t in tqdm(reversed(range(scheduler.num_train_timesteps)), desc="Denoising process"):
        t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
        noise_pred = model(x, t_tensor, label_vec)
        x = scheduler.step(noise_pred, t, x).prev_sample
        if t % 100 == 0:
            frames.append(denormalize(x.squeeze(0).cpu()))

grid = make_grid(frames, nrow=len(frames))
save_image(grid, os.path.join(save_dir, "denoise_process.png"))
