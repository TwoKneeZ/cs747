import pandas as pd
import os
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torchvision.transforms import v2
from crop_resize import crop_to_eye_region
from DRModel import DRDetection
from loss import FocalLoss
from simpleModel import simpleDRD

# script_name = os.path.splitext(os.path.basename(__file__))[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.41661593, 0.29097975, 0.20843531]
STD = [0.26398131, 0.19219237, 0.15810781]

class_weights = torch.tensor([0.272189, 2.87564, 1.3275, 8.0472, 9.922599]).to(device)
class_weights /= class_weights.sum().to(device)

guh = transforms.Lambda(crop_to_eye_region)

# Custom class for dataset loading
class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None, inverted=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.inverted = inverted  # New column for inversion status
    def __len__(self):
        return len(self.file_paths)
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = torch.load(img_path + '.pt', weights_only=True)
        
        # Check and handle inversion
        if self.inverted is not None and self.inverted[idx] == 1:
            image = v2.functional.horizontal_flip(image)  # Un-invert the image
            image = v2.functional.vertical_flip(image)        
        if self.transform:
            image = self.transform(image)
            image = guh(image)
        if self.labels is not None:  # For training and validation
            label = self.labels[idx]
            return image, label
        else:  # For testing
            return image


train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=0.5),                    # Random horizontal flip
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Random brightness/contrast
            transforms.Resize((768, 768)),
            # transforms.ToTensor(),                                    # Convert to Tensor
            transforms.Normalize(mean=MEAN,         
                                 std=STD),
        ])
val_test_transform = transforms.Compose([
            transforms.Resize((768, 768)),
            # transforms.ToTensor(),
            transforms.Normalize(mean=MEAN,
                                 std=STD),
        ])



# Load CSV
csv_path = "trainLabels_inverted.csv"
data = pd.read_csv(csv_path)
x = data['level'].value_counts()
# print(type(x))
x = dict(len(data)/x)
sorted(x.keys())
[x[i] for i in sorted(x.keys())]



# Add full paths to image files
train_folder = "images/train"
pt_folder = "images/preprocessedTrain"

data['image_path'] = data['image'].apply(lambda x: os.path.join(train_folder, x))
pt = data['image'].apply(lambda x: os.path.join(pt_folder, x))
# len(data)


# Separate "left" and "right" images
left_data = data[data['image'].str.endswith('_left')]
right_data = data[data['image'].str.endswith('_right')]
# right_data

pt_left = pt[data['image'].str.endswith('_left')]
pt_right = pt[data['image'].str.endswith('_right')]

# Add inversion status
left_inverted = left_data['inverted'].values
right_inverted = right_data['inverted'].values


# Stratified split for left images
left_train_paths, left_val_paths, left_train_labels, left_val_labels, left_train_inverted, left_val_inverted = train_test_split(
    pt_left.values,
    left_data['level'].values,
    left_inverted,
    test_size=0.1,  # 10% for validation
    stratify=left_data['level'].values,
    random_state=42
)


# Stratified split for right images
right_train_paths, right_val_paths, right_train_labels, right_val_labels, right_train_inverted, right_val_inverted = train_test_split(
    pt_right.values,
    right_data['level'].values,
    right_inverted,
    test_size=0.1,  # 10% for validation
    stratify=right_data['level'].values,
    random_state=42
)
# [1 if f.endswith('_right') else 0 for f in right_train_paths].count(1)
# Loading the dataset

# Create datasets for left and right images
left_train_dataset = CustomImageDataset(left_train_paths, left_train_labels, transform=train_transform, inverted=left_train_inverted)
left_val_dataset = CustomImageDataset(left_val_paths, left_val_labels, transform=val_test_transform, inverted=left_val_inverted)
right_train_dataset = CustomImageDataset(right_train_paths, right_train_labels, transform=train_transform, inverted=right_train_inverted)
right_val_dataset = CustomImageDataset(right_val_paths, right_val_labels, transform=val_test_transform, inverted=right_val_inverted)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    # Create data loaders for left and right images
    batch_size = 16
    left_train_loader = DataLoader(left_train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    left_val_loader = DataLoader(left_val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    right_train_loader = DataLoader(right_train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    right_val_loader = DataLoader(right_val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)


    # Model building
    def train_model_with_validation(device, model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
        model.train()
        training_loss, validation_loss = [], []
        training_acc, validation_acc = [], []
        kappa_score = []
        for epoch in range(num_epochs):
            # Training phase
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in tqdm(train_loader):
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_accuracy = 100 * correct / total
            train_loss = running_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    # Collect predictions and true labels for Kappa calculation
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
            val_accuracy = 100 * val_correct / val_total
            val_loss /= len(val_loader)

            # Quadratic Weighted Kappa Score
            kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            training_acc.append(train_accuracy)
            validation_acc.append(val_accuracy)
            kappa_score.append(kappa)

            # Print epoch metrics
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Kappa Score: {kappa:.4f}\n")
            model.train()  # Switch back to training mode
            
        return training_loss, validation_loss, training_acc, validation_acc, kappa_score

    # Model for "left" images
    epochs = 30
    left_model = simpleDRD().to(device)
    # num_features = left_model.classifier[1].in_features  # Get the input features of the classifier
    # left_model.classifier[1] = nn.Linear(num_features, 5)

    left_model = left_model.to(device)
    left_optimizer = optim.Adam(left_model.parameters(), lr=1e-1)
    left_criterion = nn.CrossEntropyLoss(weight=class_weights)

    print("\nTraining Left Model...")
    left_training_loss, left_validation_loss, left_training_acc, left_validation_acc, left_kappa_ascore = train_model_with_validation(device, left_model, left_train_loader, left_val_loader, left_optimizer, left_criterion, epochs)


    # Save the "left" model
    torch.save(left_model.state_dict(), f"model/crop_DRDetection_Inverting_left_model.pth")

    # Model for "right" images
    epochs=30
    right_model = simpleDRD().to(device)
    # num_features = right_model.classifier[1].in_features
    # right_model.classifier[1] = nn.Linear(num_features, 5)
    
    right_model = right_model.to(device)
    right_optimizer = optim.Adam(right_model.parameters(), lr=1e-1)
    right_criterion = nn.CrossEntropyLoss(weight=class_weights)


    print("\nTraining Right Model...")
    right_training_loss, right_validation_loss, right_training_acc, right_validation_acc, right_kappa_ascore = train_model_with_validation(device, right_model, right_train_loader, right_val_loader, right_optimizer, right_criterion, epochs)


    # Save the "right" model
    torch.save(right_model.state_dict(), f"model/crop_DRDetection_Inverting_right_model.pth")

    # Plot Loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(right_training_loss, label='Right Training Loss', marker='o')
    plt.plot(right_validation_loss, label='Right Validation Loss', marker='o')
    plt.plot(left_training_loss, label='Left Training Loss', marker='o')
    plt.plot(left_validation_loss, label='Left Validation Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()



    # Plot Accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(right_training_acc, label='Right Training Accuracy', marker='o')
    plt.plot(right_validation_acc, label='Right Validation Accuracy', marker='o')
    plt.plot(left_training_acc, label='Left Training Accuracy', marker='o')
    plt.plot(left_validation_acc, label='Left Validation Accuracy', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'images/crop_DRDetection_Inverting_test_val.png')


    # Plot Kappa Score
    plt.figure(figsize=(6, 4))
    plt.plot(right_kappa_ascore, label='Right Kappa Score', marker='o')
    plt.plot(left_kappa_ascore, label='Left Kappa Score', marker='o')
    plt.title('Kappa Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Kappa Score')
    plt.legend()
    plt.grid()
    plt.savefig(f'images/crop_DRDetection_Inverting_kappa.png')


    # left_model = models.efficientnet_b4().to(device)
    
    # num_features = left_model.classifier[1].in_features
    # left_model.classifier[1] = nn.Linear(num_features, 5)
    # left_model.load_state_dict(torch.load("model/efficient_net_Inverting_left_model.pth", weights_only=True))
    # left_model = left_model.to(device)


    # right_model = models.efficientnet_b4().to(device)
    
    # num_features = right_model.classifier[1].in_features
    # right_model.classifier[1] = nn.Linear(num_features, 5)
    # right_model.load_state_dict(torch.load("model/efficient_net_Inverting_right_model.pth", weights_only=True))
    # right_model = right_model.to(device)

    # test file prediction and generation
    test_folder = "images/preprocessedTest"
    test_files = os.listdir(test_folder)
    left_results = []
    right_results = []
    left_model.eval()
    right_model.eval()
    
    invertLabels = pd.read_csv('InvertedLabelsTestDataset.csv')
        
    for f in tqdm([file for file in test_files if file.endswith("_left.pt")]):
        img = torch.load(os.path.join(test_folder, f), weights_only=True)
        
        val = invertLabels.loc[invertLabels['image'].str.match(f.split('.')[0])]['inverted'].item()
        
        with torch.no_grad():
            
            if val == 1:
                img = v2.functional.horizontal_flip(img)
                img = v2.functional.vertical_flip(img)
                
            img = guh(img)
            
            outputs = left_model(val_test_transform(img).unsqueeze(0).to(device))
            _, pred = torch.max(outputs, 1)  # Get the class index with the highest score
            left_results.append([os.path.splitext(f)[0], pred.item()])
            
            
    for f in tqdm([file for file in test_files if file.endswith("_right.pt")]):
        
        
        img = torch.load(os.path.join(test_folder, f), weights_only=True)
        
        val = invertLabels.loc[invertLabels['image'].str.match(f.split('.')[0])]['inverted'].item()

        with torch.no_grad():
            
            if val == 1:
                img = v2.functional.horizontal_flip(img)
                img = v2.functional.vertical_flip(img)
            
            img = guh(img)
            
            outputs = right_model(val_test_transform(img).unsqueeze(0).to(device))
            _, pred = torch.max(outputs, 1)  # Get the class index with the highest score
            right_results.append([os.path.splitext(f)[0], pred.item()])
            
            
            
    # Combine results
    combined_results = left_results + right_results



    # Save to a single CSV
    df = pd.DataFrame(combined_results, columns=["image", "level"])
    df.to_csv(f"output/cropping_DRDetection_Inverting_predictions.csv", index=False)
    print("Predictions saved")
    