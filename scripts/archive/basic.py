import pandas as pd
import os
from PIL import Image
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
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# In[2]:


MEAN = [0.41661593, 0.29097975, 0.20843531]
STD = [0.26398131, 0.19219237, 0.15810781]


# # Custom class for dataset loading

# In[3]:


class CustomImageDataset(Dataset):
    def __init__(self, file_paths, labels=None, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path+'.jpeg').convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.labels is not None:  # For training and validation
            label = self.labels[idx]
            return image, label
        else:  # For testing
            return image


# In[4]:


train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),                    # Random horizontal flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Random brightness/contrast
            transforms.ToTensor(),                                    # Convert to Tensor
            transforms.Normalize(mean=MEAN,          # Normalize to ImageNet stats
                                 std=STD),
        ])

val_test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN,
                                 std=STD),
        ])


# In[5]:


# Load CSV
csv_path = "trainLabels.csv"
data = pd.read_csv(csv_path)

# Add full paths to image files
train_folder = "train_crop_224/"
data['image_path'] = data['image'].apply(lambda x: os.path.join(train_folder, x))

# Stratified split for training and validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    data['image_path'].values,
    data['level'].values,
    test_size=0.2,  # 20% for validation
    stratify=data['level'].values,
    random_state=42
)


# # Loading the dataset

# In[6]:


# Training and Validation datasets
train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
val_dataset = CustomImageDataset(val_paths, val_labels, transform=val_test_transform)

# Test dataset
# test_folder = "path/to/test/folder"
# test_paths = [os.path.join(test_folder, fname) for fname in os.listdir(test_folder)]
# test_dataset = CustomImageDataset(test_paths, transform=val_test_transform)

# Data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Model building

# In[15]:


def train_model_with_validation(model, train_loader, val_loader, num_epochs=10):
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


# In[18]:


model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final layer for 5 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)

criterion = nn.CrossEntropyLoss()  # You can pass class weights here if needed
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# In[19]:


epochs=50
training_loss, validation_loss, training_acc, validation_acc, kappa_score = train_model_with_validation(model, train_loader, val_loader, epochs)
torch.save(model.state_dict(), "model/basic_model.pth")


# In[21]:


# Plot Loss
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(training_loss, label='Training Loss', marker='o')
plt.plot(validation_loss, label='Validation Loss', marker='o')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot Accuracy
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(training_acc, label='Training Accuracy', marker='o')
plt.plot(validation_acc, label='Validation Accuracy', marker='o')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig('images/basic_test_val.png')

# Plot Kappa Score
plt.figure(figsize=(6, 4))
plt.plot(kappa_score, label='Kappa Score', marker='o', color='purple')
plt.title('Kappa Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Kappa Score')
plt.legend()
plt.grid()
plt.savefig('images/basic_kappa.png')

