import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

def create_directory(directory):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def crop_to_eye_region(image_path, output_size=(512, 512)):
    """
    Crop the retina fundus image to focus on the circular eye region and remove black borders.
    
    Args:
        image_path (str): Path to the input image.
        output_size (tuple): The desired output size (height, width) for resizing.
    
    Returns:
        PIL.Image.Image: Cropped and resized image.
    """
    # Load the image
    img = cv2.imread(image_path)
    # img = cv2.cuda_GpuMat().upload(img)
    
    # CLAHE
    # img = cv2.fastNlMeansDenoisingColored(img, None, 1, 1, 7, 21)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Threshold to segment the bright retina region
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour, assuming it's the retina
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Rescale if w:h ratio is smaller than average (1.31)
    # Round down if needed
    if w/h < 1.31:
        new_h = int(w / 1.31)
        new_y = y + int((h - new_h) / 2)
        y, h = new_y, new_h
        
    # Crop the image to the bounding box
    img = img[y:y+h, x:x+w]

    # # Crop the tab
    # mask = np.zeros_like(img)
    # mask = cv2.circle(mask, (int(w/2), int(h/2)), int(0.48*w), (255,255,255), -1)
    # img[mask==0] = 0
    
    
    # Resize the cropped image to the desired size
    resized_img = cv2.resize(img, output_size, interpolation=cv2.INTER_AREA)
    
    # Convert to PIL Image for further processing
    return Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))

def process_image(input_path, output_path, resize_size):
    """
    Process a single image: crop and resize it.
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image.
        resize_size (int): Target size for resizing (height, width).
    """
    
    # Process the image
    processed_img = crop_to_eye_region(input_path, output_size=(resize_size, resize_size))
    # Save the processed image
    processed_img.save(output_path)
    
def process_images(input_dir, output_dir, resize_size):
    """
    Apply the transformations to all images in the directory.
    Args:
        input_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the processed images.
        crop_size (tuple): Crop size (height, width).
        resize_size (int): Target size for resizing (height, width).
    """
    create_directory(output_dir)

    # List all images in the directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpeg'))]

#     for image_file in tqdm(image_files):
#         input_path = os.path.join(input_dir, image_file)
#         output_path = os.path.join(output_dir, image_file)

#         process_image(input_path, output_path, resize_size)
        
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for image_file in image_files:
            input_path = os.path.join(input_dir, image_file)
            output_path = os.path.join(output_dir, image_file)
            futures.append(executor.submit(process_image, input_path, output_path, resize_size))

        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                future.result()  # Wait for the task to complete
                pbar.update(1)  # Update the progress bar when each future completes

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process images by resizing and saving to a specific directory.")
    
    # Define command-line arguments
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed images.")
    parser.add_argument("--resize_size", type=int, required=True, help="Size to which images should be resized.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with command-line arguments
    process_images(args.input_dir, args.output_dir, args.resize_size)

