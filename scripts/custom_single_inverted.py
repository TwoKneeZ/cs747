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
from model import DRDetection
from loss import FocalLoss

script_name = os.path.splitext(os.path.basename(__file__))[0]


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
        # self.inverted = inverted  # New column for inversion status

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path + '.jpeg').convert("RGB")
        
        # # Check and handle inversion
        # if self.inverted is not None and self.inverted[idx] == 1:
        #     image = ImageOps.flip(image)  # Un-invert the image

        if self.transform:
            image = self.transform(image)

        if self.labels is not None:  # For training and validation
            label = self.labels[idx]
            return image, label
        else:  # For testing
            return image


# In[4]:


train_transform = transforms.Compose([
            # transforms.Resize((380, 380)),
            # transforms.RandomHorizontalFlip(p=0.5),                    # Random horizontal flip
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),      # Random brightness/contrast
            transforms.ToTensor(),                                    # Convert to Tensor
            transforms.Normalize(mean=MEAN,         
                                 std=STD),
        ])

val_test_transform = transforms.Compose([
            # transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN,
                                 std=STD),
        ])


# In[5]:


# Load CSV
csv_path = "train_oversampled.csv"
data = pd.read_csv(csv_path)

class_weights = dict(len(data)/data['level'].value_counts())
class_weights = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float)

# Add full paths to image files
train_folder = "train_oversampled/"
data['image_path'] = data['image'].apply(lambda x: os.path.join(train_folder, x))

# inverted = data['inverted'].values

# Stratified split for training and validation
train_paths, val_paths, train_labels, val_labels = train_test_split(
    data['image_path'].values,
    data['level'].values,
    # inverted,
    test_size=0.1,  # 10% for validation
    stratify=data['level'].values,
    random_state=42
)


# # Loading the dataset

# In[6]:


# Training and Validation datasets
train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform)
val_dataset = CustomImageDataset(val_paths, val_labels, transform=val_test_transform)

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# # Model building

# In[15]:


def train_model_with_validation(device, model, train_loader, val_loader, optimizer, criterion, scheduler=None, num_epochs=10):
    model.train()
    training_loss, validation_loss = [], []
    training_acc, validation_acc = [], []
    kappa_score = []
    for epoch in range(num_epochs):
        # Training phase
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images) #inceptionv3 returns two outputs
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
            for images, labels in val_loader:
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

        if scheduler:
            scheduler.step()

        # Print epoch metrics
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Kappa Score: {kappa:.4f}\n")

        model.train()  # Switch back to training mode
        
    return training_loss, validation_loss, training_acc, validation_acc, kappa_score

def init_model():
    # model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    # num_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_features, 5)
    model = DRDetection(dropout_prob=0.5, leakiness=0.01)
    
    return model

# In[18]:

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = init_model().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))  # You can pass class weights here if needed
# criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# In[19]:


epochs=30
training_loss, validation_loss, training_acc, validation_acc, kappa_score = train_model_with_validation(device, model, train_loader, val_loader, optimizer, criterion, scheduler, epochs)
torch.save(model.state_dict(), f"model/{script_name}_model.pth")


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
plt.savefig(f'images/{script_name}_test_val.png')

# Plot Kappa Score
plt.figure(figsize=(6, 4))
plt.plot(kappa_score, label='Kappa Score', marker='o', color='purple')
plt.title('Kappa Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Kappa Score')
plt.legend()
plt.grid()
plt.savefig(f'images/{script_name}_kappa.png')



test_folder = "test_prep512/"
test_files = os.listdir(test_folder)

results = []

model.eval()


for f in test_files:
    img = Image.open(os.path.join(test_folder, f)).convert('RGB')
    with torch.no_grad():
        outputs = model(val_test_transform(img).unsqueeze(0).to(device))
        _, pred = torch.max(outputs, 1)  # Get the class index with the highest score
        results.append([os.path.splitext(f)[0], pred.item()])


# Save to a single CSV
df = pd.DataFrame(results, columns=["image", "level"])
df.to_csv(f"output/{script_name}_predictions.csv", index=False)
print("Predictions saved")