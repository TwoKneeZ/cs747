import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Crop the eye image (with w:h = 1.31), and remove the tab
def imageCrop(img):
    # Load image to cv2
    
    img = img.permute(1, 2, 0).numpy()
    
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    img = (img*255).astype('uint8')
    
    # Contrast Limited Adaptive Histogram Equalization to deal with different image lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Find the largest contour
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Rescale if w:h ratio is smaller than average (1.31)
    # Round down if needed
    if w/h < 1.31:
        new_h = int(w / 1.31)
        new_y = y + int((h - new_h) / 2)
        y, h = new_y, new_h
    
    # Crop the image
    img = img[y:y+h, x:x+w]

    # Crop the tab
    mask = np.zeros_like(img)
    mask = cv2.circle(mask, (int(w/2), int(h/2)), int(0.45*w), (255,255,255), -1)
    img[mask==0] = 0
    
    # Contrast enhancement
    brightness = -60
    contrast = 1.5
    img = cv2.addWeighted(img, contrast, img, 0, brightness)
    
    # Image sharpening
    blur = cv2.GaussianBlur(img,(11, 11),0)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    
    # Apply color map (convert to RGB for processing)
    cm = plt.get_cmap('gray')
    img = cm(img)
    return torch.from_numpy(img[:, :, :3]).permute(2, 0, 1).to(dtype=torch.float32)