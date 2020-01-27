## TODO: define the convolutional neural network architecture
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def init_weights(op):
    if isinstance(op, nn.Conv2d) or isinstance(op, nn.Linear):
        I.xavier_uniform_(op.weight)
    if isinstance(op, nn.Linear):
        op.bias.data.fill_(0.01)

is_debug = False

def print_layers(net):
    global is_debug
    is_debug = True
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

        self.weight = self.conv.weight.data

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
        self.conv1 = ConvBatch2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = ConvBatch2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = ConvBatch2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = ConvBatch2d(64, 128, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = ConvBatch2d(128, 256, 3)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.dropout1 = nn.Dropout2d(p=0.1)

        self.conv6 = ConvBatch2d(256, 512, 3)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.dropout2 = nn.Dropout2d(p=0.2)

        self.fc1 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(p=0.4)

        self.fc2 = nn.Linear(256, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.conv1(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.conv1._get_name(), x.size()))
            print('')
        x = self.pool1(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.pool1._get_name(), x.size()))
            print('')

        x = self.conv2(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.conv2._get_name(), x.size()))
            print('')
        x = self.pool2(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.pool2._get_name(), x.size()))
            print('')

        x = self.conv3(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.conv3._get_name(), x.size()))
            print('')

        x = self.pool3(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.pool3._get_name(), x.size()))
            print('')

        x = self.conv4(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.conv4._get_name(), x.size()))
            print('')

        x = self.pool4(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.pool4._get_name(), x.size()))
            print('')

        x = self.conv5(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.conv5._get_name(), x.size()))
            print('')

        x = self.pool5(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.pool5._get_name(), x.size()))
            print('')

        x = self.dropout1(x)

        x = self.conv6(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.conv6._get_name(), x.size()))
            print('')

        x = self.pool6(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.pool6._get_name(), x.size()))
            print('')

        x = self.dropout2(x)

        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))

        if is_debug:
            print('')
            print('{}: {}'.format('Flatten', x.size()))
            print('')

        x = F.leaky_relu(self.fc1(x))
        if is_debug:
            print('')
            print('{}: {}'.format(self.fc1._get_name(), x.size()))
            print('')
        x = self.dropout3(x)

        x = self.fc2(x)
        if is_debug:
            print('')
            print('{}: {}'.format(self.fc2._get_name(), x.size()))
            print('')

        # a modified x, having gone through all the layers of your model, should be returned
        return x

if __name__ == '__main__':
    net = Net()

    print_layers(net)
