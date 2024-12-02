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
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from InversionClassifier import InversionClassifier
from torchvision.transforms import v2
from simpleModel import simpleDRD
from utils import imageCrop

MEAN = [0.41661593, 0.29097975, 0.20843531]
STD = [0.26398131, 0.19219237, 0.15810781]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = torch.tensor([0.272189, 2.87564, 1.3275, 8.0472, 9.922599]).to(device)
class_weights /= class_weights.sum().to(device)

guh = transforms.Lambda(imageCrop)

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
            
        image = guh(image)
                
        # Check and handle inversion
        if self.inverted is not None and self.inverted[idx] == 1:
            image = v2.functional.horizontal_flip(image)  # Un-invert the image
            image = v2.functional.vertical_flip(image)
            
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:  # For training and validation
            label = self.labels[idx]
            return image, label
        else:  # For testing
            return image

train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.5),     
            # transforms.RandomRotation(360),
            # Random horizontal flip
            transforms.ColorJitter(brightness=0.05, contrast=0.05),      # Random brightness/contrast
            # transforms.ToTensor(),                                    # Convert to Tensor
            # transforms.Normalize(mean=MEAN,          # Normalize to ImageNet stats
            #                      std=STD),
        ])

val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=MEAN,
            #                      std=STD),
        ])

# image_folder = "images/test/test"
# output_folder = "images/preprocessedTest"

# os.makedirs(output_folder, exist_ok=True)

# for filename in os.listdir(image_folder):
#     if filename.endswith(".jpeg"):
        
#         image_path = os.path.join(image_folder, filename)
#         image = Image.open(image_path).convert("RGB")
        
#         tensor = train_transform(image)
        
#         save_path = os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.pt')
        
#         torch.save(tensor, save_path)


# print("done")

# Load CSV
csv_path = "trainLabels_inverted.csv"
data = pd.read_csv(csv_path)

x = data['level'].value_counts()
x = dict(len(data)/x)
sorted(x.keys())
[x[i] for i in sorted(x.keys())]


# Add full paths to image files
train_folder = "images/train"
pt_folder = "images/preprocessedTrain"

data['image_path'] = data['image'].apply(lambda x: os.path.join(train_folder, x))
pt = data['image'].apply(lambda x: os.path.join(pt_folder, x))


#Check if needed
invertedVals = data['inverted'].values


# Stratified split for training and validation
train_paths, val_paths, train_labels, val_labels, train_inverted, val_inverted = train_test_split(
    pt.values,
    data['level'].values,
    invertedVals,
    test_size=0.2,  # 20% for validation
    stratify=data['level'].values,
    random_state=42
)


# Training and Validation datasets

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    
    # Loading the dataset
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

            if kappa > max(kappa_score):
                torch.save(model.state_dict(), "model/bestkappa.pth")


            model.train()  # Switch back to training mode
            
        return training_loss, validation_loss, training_acc, validation_acc, kappa_score


    train_dataset = CustomImageDataset(train_paths, train_labels, transform=train_transform, inverted=train_inverted)
    val_dataset = CustomImageDataset(val_paths, val_labels, transform=val_test_transform, inverted=val_inverted)

    # Data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    
    model = simpleDRD().to(device)

    # # Modify the final layer for 5 classes
    # num_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_features, 5)

    criterion = nn.CrossEntropyLoss(weight=class_weights)  # You can pass class weights here if needed
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # # Move model to GPU if available
    model = model.to(device)
    
    epochs=30
    training_loss, validation_loss, training_acc, validation_acc, kappa_score = train_model_with_validation(device, model, train_loader, val_loader, optimizer, criterion, epochs)
    

    torch.save(model.state_dict(), "model/preproc.pth")
    # # Plot Loss
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
    plt.savefig('images/preproc_acc.png')

    # Plot Kappa Score
    plt.figure(figsize=(6, 4))
    plt.plot(kappa_score, label='Kappa Score', marker='o', color='purple')
    plt.title('Kappa Score over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Kappa Score')
    plt.legend()
    plt.grid()
    plt.savefig('images/preproc_kappa.png')


    # inversionModel = InversionClassifier().to(device)
    # inversionModel.load_state_dict(torch.load("InversionModel", weights_only=True))
    # inversionModel.eval()
    
    
    # model = simpleDRD()
    # model.load_state_dict(torch.load("model/bestkappa.pth", weights_only=True))
    
    # num_features = model.classifier[1].in_features
    # model.classifier[1] = nn.Linear(num_features, 5)
    # model.load_state_dict(torch.load("model/test.pth", weights_only=False))


    # 2. Prepare the test images dataset
    test_folder = "images/preprocessedTest"
    test_files = os.listdir(test_folder)
    results = []
    # model = model.to(device)
    model.eval()
    
    invertLabels = pd.read_csv('InvertedLabelsTestDataset.csv')

    for f in tqdm(test_files, miniters=1000):
        
        img = torch.load(os.path.join(test_folder, f), weights_only=True)
    
        val = invertLabels.loc[invertLabels['image'].str.match(f.split('.')[0])]['inverted'].item()
        
        # img = Image.open(os.path.join(test_folder, f)).convert('RGB')
        with torch.no_grad():
            
            # inversionOutput = inversionModel(val_test_transform(img).unsqueeze(0).to(device))
            
            # print(inversionOutput, f)
            
            # isInverted = torch.argmax(inversionOutput)
            
            img = guh(img)
            
            if val == 1:
                img = v2.functional.horizontal_flip(img)
                img = v2.functional.vertical_flip(img)
                            
            outputs = model(val_test_transform(img).unsqueeze(0).to(device))
            _, pred = torch.max(outputs, 1)  # Get the class index with the highest score
            results.append([os.path.splitext(f)[0], pred.item()])

    # 4. Save to CSV
    df = pd.DataFrame(results, columns=["image", "level"])
    df.to_csv("output/preproc.csv", index=False)
