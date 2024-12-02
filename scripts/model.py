import torch
import torch.nn as nn
import torch.nn.functional as F

class DRDetection(nn.Module):
    def __init__(self, dropout_prob= 0.5, leakiness=0.5):
        super(DRDetection, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        #max_pool2d(kernel=3, stride=2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        #max_pool2d
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        #maxpool2d
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=leakiness),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        #max_pool2d
        self.layer12 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=leakiness)
        )

        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.Dropout(p=dropout_prob),
            nn.Linear(1024, 512),
            nn.Dropout(p=dropout_prob),
            nn.Linear(512,5)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, 1)
                nn.init.constant_(m.bias, 0.05)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x) 
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
