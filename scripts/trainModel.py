import torch
from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.metrics import cohen_kappa_score
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def loader(path):
    return torch.load(path, weights_only=True)

specific = 'left'
def isValid(path):
    return specific in path

if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn", force=True)
        
    batch_size = 32
    root = './tester'

    images = DatasetFolder(
        root=root,
        loader=loader,
        extensions=('.pt',),
        # is_valid_file=isValid
    )

    labels = [label for _, label in images.samples]
    train_indices, val_indices = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            stratify=labels
        )

    train_dataset = Subset(images, train_indices)
    val_dataset = Subset(images, val_indices)

    train_labels = [labels[i] for i in train_indices]
    val_labels = [labels[i] for i in val_indices]
    print("Training class counts:", np.bincount(train_labels))
    print("Validation class counts:", np.bincount(val_labels))

    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float) 
    sample_weights = [class_weights[labels[i]] for i in train_indices]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True 
    )


    # train_size = int(0.8 * len(images))
    # val_size = len(images) - train_size
    # train_dataset, test_dataset = random_split(images, [train_size, val_size])
    # print(train_size, val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True, 
        shuffle=False, 
        persistent_workers=True,
        sampler=sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        persistent_workers=True
    )

    from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    from models import DRD, simpleDRD

    model = DRD()
    # model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)
    # model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model = model.to(device)

    from loss import FocalLoss
    criterion = FocalLoss(gamma=2).to(device)
    # criterion = DifferentiableQWKLoss(num_classes=5).to(device)
    # criterion = WeightedKappaLoss(num_classes=5, device=device, regression=False)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, fused=True, momentum=0.9, weight_decay=1e-4)
    # optimizer = optim.Adam(model.parameters(), lr=1e-3, fused=True)
    num_epochs = 5000

    best = 0.4
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')
        
        trainPreds = []
        trainLabels = []
        trainTotal = 0
        trainCorrect = 0.0
        trainLoss = 0.0
        
        model.train()
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            output = model(images)
            pred = torch.argmax(output, dim=1)
    
            loss = criterion(output, labels)
            
            trainTotal += labels.size(0)
            trainCorrect += (pred == labels).sum().item()
            
            trainPreds.extend(pred.cpu().numpy())
            trainLabels.extend(labels.cpu().numpy())
            
            loss.backward()
            optimizer.step()
            
            trainLoss += loss.item()
        
        valPreds = []
        valLabels = []
        valCorrect = 0
        valTotal = 0
        valLoss = 0.0
        
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader):   
                images, labels = images.to(device), labels.to(device)
                
                output = model(images)
                pred = torch.argmax(output, dim=1)
                
                loss = criterion(output, labels)
                
                valTotal += labels.size(0)
                valCorrect += (pred == labels).sum().item()
                
                valPreds.extend(pred.cpu().numpy())
                valLabels.extend(labels.cpu().numpy())
                
                valLoss += loss

        trainQWK = cohen_kappa_score(trainPreds, trainLabels, weights='quadratic')
        valQWK = cohen_kappa_score(valPreds, valLabels, weights='quadratic')
        
        print(f'Training       Loss: {trainLoss / len(train_loader)}')
        print(f'Validation     Loss: {valLoss / len(val_loader)}')
        
        print(f'Training   Accuracy: {100 * trainCorrect / trainTotal}%')
        print(f'Validation Accuracy: {100 * valCorrect / valTotal}%')
        
        print(f'Training        QWK: {trainQWK}')
        print(f"Validation      QWK: {valQWK}")
        
        if valQWK >= best:
            torch.save(model, f'models/Classifier-{valQWK:.6f}') 
            best = valQWK