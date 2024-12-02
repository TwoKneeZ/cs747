import os
import random
from PIL import Image

# Specify the directory containing images
image_dir = "train_crop_224/"

# Get a list of all files in the directory
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg','.jpeg', '.bmp', '.gif'))]

if not image_files:
    print("No images found in the specified directory.")
else:
    # Pick a random image file
    random_image = random.choice(image_files)
    print(f"Opening image: {random_image}")

    # Open the image using PIL
    image_path = os.path.join(image_dir, random_image)
    image = Image.open(image_path)

    # Show the image
    image.show()
