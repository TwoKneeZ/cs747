import torch
import torch.nn as nn

NUM_CLASSES = 5

class simpleDRD(nn.Module):
    def __init__(self):
        super(simpleDRD, self).__init__()
        
        self.activateFn = nn.LeakyReLU(negative_slope=.01)
        self.dropout = nn.Dropout(p=0.3)

class simpleDRD(nn.Module):
    def __init__(self):
        super(simpleDRD, self).__init__()
        
        self.activateFn = nn.LeakyReLU(negative_slope=.01)
        self.dropout = nn.Dropout(p=0.3)
             
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn5 = nn.BatchNorm2d(256)
        # self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.bn6 = nn.BatchNorm2d(256)
        # self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.bn7 = nn.BatchNorm2d(512)
        # self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.bn8 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d((2, 2))
        self.finPool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(256, NUM_CLASSES)
        )       
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.bn1(x)
        x = self.activateFn(x)
        
        x = self.conv2(x)
        x = self.pool(x)
        x = self.bn2(x)
        x = self.activateFn(x)
        
        x = self.conv3(x)
        x = self.pool(x)
        x = self.bn3(x)
        x = self.activateFn(x)
        
        x = self.conv4(x)
        x = self.pool(x)
        x = self.bn4(x)
        x = self.activateFn(x)
        
        # x = self.conv5(x)
        # x = self.pool(x)
        # x = self.bn5(x)
        # x = self.activateFn(x)
        
        # x = self.conv6(x)
        # x = self.pool(x)
        # x = self.bn6(x)
        # x = self.activateFn(x)
        
        # x = self.conv7(x)
        # x = self.pool(x)
        # x = self.bn7(x)
        # x = self.activateFn(x)
        
        # x = self.conv8(x)
        # x = self.pool(x)
        # x = self.bn8(x)
        # x = self.activateFn(x)
        
        x = self.finPool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
