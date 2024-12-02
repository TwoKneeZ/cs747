import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from tqdm import tqdm

# Define paths
image_dir = './0'
output_dir = "./tester/0"
os.makedirs(output_dir, exist_ok=True)

from albumentations import CLAHE, Compose
transform = Compose([CLAHE(clip_limit=4, tile_grid_size=(8, 8), always_apply=1.0)])
def toCLAHE(img):
    img = np.array(img)
    img = transform(image=img)['image']
    return Image.fromarray(img)


torch_transform = transforms.Compose([
    transforms.Lambda(toCLAHE),
    transforms.Resize((256, 256)),  # Resize all images to the same size
    transforms.ToTensor()          # Convert image to tensor
])

# Convert and save each image
for image_file in tqdm(os.listdir(image_dir)):
    if image_file.endswith(".jpeg"):
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
        tensor_image = torch_transform(image)
        
        # Save as .pt
        output_path = os.path.join(output_dir, image_file.replace(".jpeg", ".pt"))
        torch.save(tensor_image, output_path)