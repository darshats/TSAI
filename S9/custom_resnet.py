from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.prepBlock = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        ## 30
        self.maxpool1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        ## 15
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential( 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        #13
        
        self.maxpool2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.MaxPool2d(2,2),
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        #6
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(num_features=512, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        
        self.maxpool3 = nn.MaxPool2d(4,4)
        #1
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.prepBlock(x) 
        
        x = self.maxpool1(x)
        resx_1 = self.layer1(x)
        x = torch.add(x, resx_1)  ## 128 channels
        
        x = self.layer2(x) 
        
        x = self.maxpool2(x)
        resx_2 = self.layer3(x)
        x = torch.add(x, resx_2)
        
        x = self.maxpool3(x)
        x = x.view(-1, 512)
        
        x = self.fc(x)
        return F.log_softmax(x)
