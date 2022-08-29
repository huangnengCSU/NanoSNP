import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self,X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y+X)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

def ResNet(input_channels, num_classes):
    # b1 = nn.Sequential(nn.Conv2d(5,64,kernel_size=5,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=2,stride=2))  # [N x 64 x 39 x 4]
    b1 = nn.Sequential(nn.Conv2d(input_channels,64,kernel_size=5,stride=1,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))  # [N x 64 x 39 x 5]

    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))  # [N x 64 x 39 x 5]
    b3 = nn.Sequential(*resnet_block(64, 128, 2))   # [N x 128 x 20 x 3]
    b4 = nn.Sequential(*resnet_block(128, 256, 2))  # [N x 256 x 10 x 2]
    b5 = nn.Sequential(*resnet_block(256, 512, 2)) # [N x 512 x 5 x 1]

    net = nn.Sequential(b1,b2,b3,b4,b5,nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(512,num_classes))  # [N x 10]
    return net

if __name__=="__main__":
    net = ResNet(4,128)
    # X = torch.randn(size=(128, 20, 20, 11))
    X = torch.randn(size=(128, 4, 60, 11))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)