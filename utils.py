import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_images(images, color=False):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        if color:
            plt.imshow(np.swapaxes(np.swapaxes(img, 0, 1), 1, 2))
        else:
            plt.imshow(np.swapaxes(np.swapaxes(img, 0, 1), 1, 2), cmap='gray')
    return 

def blackAndWhite(img):
    min_val = img.min()
    
    mask = (img <= min_val + 0.03).all(dim=0)
    img[:, ~mask] = 1.0
    
    return img

# Crop the eye image (with w:h = 1.31), and remove the tab
def grayscaleAndCrop(img):
    # Load image to cv2
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Contrast Limited Adaptive Histogram Equalization to deal with different image lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Find the largest contour
    _, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

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
    mask = cv2.circle(mask, (int(w/2), int(h/2)), int(0.48*w), (255,255,255), -1)
    img[mask==0] = 0
    
    # Convert to PIL Image for further processing
    return Image.fromarray(img)