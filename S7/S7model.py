# from S7model import Net
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ## depthwise convolution step 1 -- use 3 3x3 kernels. Also using dilation
        self.convblock1 = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, dilation=2),
            nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        
        ## depthwise convolution step 2 -- use 1x1 kernels
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )

        ## strided convolution to bring down size to half ~13x13
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        
        ## dilated convolutions take size from 13-->9-->5
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, dilation=2),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, dilation=2),
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1),
            nn.ReLU()
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.convblock6 = nn.Conv2d(in_channels=64, out_channels=10, kernel_size=1)
        #self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.convblock1(x) # o 16, 28, 28. RF 5
        x = self.convblock2(x) #o 32, 28, 28. RF 7
        x = self.convblock3(x) #o 32, 13, 13. RF 15

        x = self.convblock4(x) #o 64, 9, 9. RF 23
        x = self.convblock5(x) #o 64, 5, 5. RF 31

        x = self.gap(x) #o 64, 1, 1. RF 31
        x = self.convblock6(x) #o 10, 1, 1. RF 31
        x = x.view(-1, 10)
        return x
