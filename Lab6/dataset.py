import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Load object index mapping (24 possible objects: 3 shapes × 8 colors)
with open('objects.json', 'r') as f:
    object_to_idx = json.load(f)
num_objects = len(object_to_idx)  # should be 24

# Define transformation for images: resize to 64x64 and normalize to [-1,1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

class ICLEVRDataset(Dataset):
    def __init__(self, json_path, img_dir=None, transform=transform):
        """
        Dataset for i-CLEVR. If img_dir is provided, images are loaded from that directory.
        If img_dir is None, we assume this dataset is for generation (no actual images, just conditions).
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.img_dir = img_dir
        self.transform = transform
        # If the JSON is a dict (train set), keys are image filenames, values are label lists.
        # If the JSON is a list (test sets), each element is a list of labels.
        if isinstance(data, dict):
            self.filenames = list(data.keys())
            self.labels_list = list(data.values())
        else:
            # For test sets, we'll treat each entry as having no filename (to be generated)
            self.filenames = [None] * len(data)
            self.labels_list = data
        self.length = len(self.labels_list)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        labels = self.labels_list[idx]  # list of object names
        # Create multi-hot label vector for this image
        multi_hot_label = torch.zeros(num_objects, dtype=torch.float)
        for obj in labels:
            if obj in object_to_idx:
                multi_hot_label[object_to_idx[obj]] = 1.0
        if self.img_dir is not None:
            # Load image from directory if available
            fname = self.filenames[idx]
            img_path = f"{self.img_dir}/{fname}"
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, multi_hot_label
        else:
            # No image, return only label (to be used for generation)
            return multi_hot_label
