import pandas as pd
import os
import cv2
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import random

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def apply_random_transformations(image):
    """Apply a random geometric transformation (rotation or mirroring) to an image using torchvision."""
    # Define a list of transformations
    transform_list = []
    if random.choice([True, False]):
        # Randomly rotate the image by 90, 180, or 270 degrees
        angle = random.choice([90, 180, 270])
        transform_list.append(transforms.RandomRotation(degrees=(angle, angle)))
    
    if random.choice([True, False]):
        # Randomly mirror the image (flip horizontally)
        transform_list.append(transforms.RandomHorizontalFlip(p=1.0))

    if random.choice([True, False]):
        # Randomly mirror the image (flip horizontally)
        transform_list.append(transforms.RandomVerticalFlip(p=1.0))
    
    # Combine transformations
    if transform_list:
        transform = transforms.Compose(transform_list)
        # Apply the transformations
        return transform(image)
    else:
        # No transformation
        return image


def copy_images(image_dir, output_dir, list_images, transform):
    for img in tqdm(list_images):
        image_file = f'{image_dir}/{img}.jpeg'
        image = Image.open(image_file).convert('RGB')
        if image is None:
            print(f"Image not found: {image_file}")
            continue
        
        # Apply a random transformation if specified
        if transform:
            transformed_image = apply_random_transformations(image)
        else:
            transformed_image = image
        
        # Save the image (transformed or original) to the output directory
        save_path = f'{output_dir}/{img}.jpeg'
        transformed_image.save(save_path)


def rotate_images(image_dir, outptut_dir, degrees_of_rotation, list_images):
    for img in tqdm(list_images):
        image_file = f"{image_dir}/{img}.jpeg"
        # Load the image
        image = cv2.imread(image_file)
        if image is None:
            print(f"Image not found: {image_file}")
            continue
        
        # Get image dimensions
        (h, w) = image.shape[:2]
        # Calculate the center of the image
        center = (w // 2, h // 2)
        # Get the rotation matrix for the specified angle
        rotation_matrix = cv2.getRotationMatrix2D(center, degrees_of_rotation, 1.0)
        # Perform the rotation
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        # Save the rotated image
        save_path = f"{outptut_dir}/{img}_{degrees_of_rotation}.jpeg"
        cv2.imwrite(save_path, rotated_image)
    
    
def mirror_images(image_dir, output_dir, list_image):
    for img in tqdm(list_image):
        image_file = f"{image_dir}/{img}.jpeg"
        # Load the image
        image = cv2.imread(image_file)
        #flip it
        image = cv2.flip(image, 1)
        save_path = f"{outptut_dir}/{img}_mirrored.jpeg"
        cv2.imwrite(save_path, image)

if __name__ == "__main__":
    trainLabels = pd.read_csv("trainLabels.csv")
    
    trainLabels_DR0 = trainLabels[trainLabels['level'] == 0]
    trainLabels_DR1 = trainLabels[trainLabels['level'] == 1]
    trainLabels_DR2 = trainLabels[trainLabels['level'] == 2]
    trainLabels_DR3 = trainLabels[trainLabels['level'] == 3]
    trainLabels_DR4 = trainLabels[trainLabels['level'] == 4]
    
    list_images_DR0 = [i for i in trainLabels_DR0['image']]
    list_images_DR1 = [i for i in trainLabels_DR1['image']]
    list_images_DR2 = [i for i in trainLabels_DR2['image']]
    list_images_DR3 = [i for i in trainLabels_DR3['image']]
    list_images_DR4 = [i for i in trainLabels_DR4['image']]
    
    input_dir = 'train_prep512/'
    outptut_dir = 'train_oversampled/'
    create_directory(outptut_dir)

    #images having dr1 just copy them
    print('copying dr1 images')
    copy_images(input_dir, outptut_dir, list_images_DR0, transform=True)
    
    
    #images having dr2
    print("processing dr2 images")
    copy_images(input_dir, outptut_dir, list_images_DR2, transform=False)
    rotate_images(input_dir, outptut_dir, 90, list_images_DR2)
    rotate_images(input_dir, outptut_dir, 120, list_images_DR2)
    
    
    #images having dr1
    print("processing dr1 images")
    copy_images(input_dir, outptut_dir, list_images_DR1, transform=False) 
    rotate_images(input_dir, outptut_dir, 90, list_images_DR1)
    rotate_images(input_dir, outptut_dir, 120, list_images_DR1)
    rotate_images(input_dir, outptut_dir, 180, list_images_DR1)
    
    #images having dr3
    print("processing dr3 images")
    copy_images(input_dir, outptut_dir, list_images_DR3, transform=False)
    rotate_images(input_dir, outptut_dir, 90, list_images_DR3)
    rotate_images(input_dir, outptut_dir, 120, list_images_DR3)
    rotate_images(input_dir, outptut_dir, 180, list_images_DR3)
    rotate_images(input_dir, outptut_dir, 270, list_images_DR3)
    mirror_images(input_dir, outptut_dir, list_images_DR3)

    #images having dr4
    print("processing dr4 images")
    copy_images(input_dir, outptut_dir, list_images_DR4, transform=False)
    rotate_images(input_dir, outptut_dir, 90, list_images_DR4)
    rotate_images(input_dir, outptut_dir, 120, list_images_DR4)
    rotate_images(input_dir, outptut_dir, 180, list_images_DR4)
    rotate_images(input_dir, outptut_dir, 270, list_images_DR4)
    mirror_images(input_dir, outptut_dir, list_images_DR4)