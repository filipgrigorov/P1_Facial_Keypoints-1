## TODO: define the convolutional neural network architecture
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def print_layers(net):
    print('\n Number of learnable parameters: ')
    nparams = 0
    for child in net.children():
        for param in child.parameters():
            if param.requires_grad:
                nparams += param.numel()
    print(' ', nparams)
    print('')

    dummy_img = torch.zeros((1, 1, 224, 224))
    net(dummy_img)

class ConvBatch2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)):
        super(ConvBatch2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch2d = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch2d(x)
        return torch.relu(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel

        #[1x224x224]
        self.conv1 = ConvBatch2d(1, 32, 3)
        #[32x222x222]
        self.conv2 = ConvBatch2d(32, 32, 3)
        #[32x220x220]
        self.pool1 = nn.MaxPool2d(2, 2)
        #[32x110x110]
        self.conv3 = ConvBatch2d(32, 64, 3)
        #[64x108x108]
        self.pool2 = nn.MaxPool2d(2, 2)
        #[64x54x54]
        self.conv4 = ConvBatch2d(64, 128, 3)
        #[128x52x52]
        self.pool3 = nn.MaxPool2d(2, 2)
        #[128x26x26]
        self.conv5 = ConvBatch2d(128, 256, 3)
        #[256x24x24]
        self.pool4 = nn.MaxPool2d(2, 2)
        #[256x12x12]
        self.conv6 = ConvBatch2d(256, 512, 3)
        #[512x10x10]
        self.pool5 = nn.MaxPool2d(2, 2)
        #[512x5x5] -> 2560
        self.fc1 = nn.Linear(2560, 1000)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(1000, 256)
        self.dropout3 = nn.Dropout(p=0.3)
        self.fc4 = nn.Linear(256, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.conv1._get_name(), x.size()))
            print('')
        x = self.conv2(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.conv2._get_name(), x.size()))
            print('')
        x = self.pool1(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.pool1._get_name(), x.size()))
            print('')
        x = self.conv3(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.conv3._get_name(), x.size()))
            print('')
        x = self.pool2(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.pool2._get_name(), x.size()))
            print('')
        x = self.conv4(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.conv4._get_name(), x.size()))
            print('')
        x = self.pool3(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.pool3._get_name(), x.size()))
            print('')
        x = self.conv5(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.conv5._get_name(), x.size()))
            print('')
        x = self.pool4(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.pool4._get_name(), x.size()))
            print('')
        x = self.conv6(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.conv6._get_name(), x.size()))
            print('')
        x = self.pool5(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.pool5._get_name(), x.size()))
            print('')

        x = x.view(-1, x.size(0) * x.size(1) * x.size(2))

        if __debug__:
            print('')
            print('{}: {}'.format('Flatten', x.size()))
            print('')

        x = self.fc1(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.fc1._get_name(), x.size()))
            print('')
        x = self.dropout1(x)
        x = self.fc2(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.fc2._get_name(), x.size()))
            print('')
        x = self.dropout2(x)
        x = self.fc3(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.fc3._get_name(), x.size()))
            print('')
        x = self.dropout3(x)
        x = self.fc4(x)
        if __debug__:
            print('')
            print('{}: {}'.format(self.fc4._get_name(), x.size()))
            print('')
        # a modified x, having gone through all the layers of your model, should be returned
        return x

if __name__ == '__main__':
    net = Net()

    print_layers(net)
