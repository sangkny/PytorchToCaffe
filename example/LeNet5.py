'''LeNet in PyTorch.

'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining the network (LeNet-5)
class LeNet5(torch.nn.Module):
    def __init__(self, num_classes=2, in_ch=3):
        super(LeNet5, self).__init__()
        # Convolution 1 (In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below)
        self.conv1 = torch.nn.Conv2d(in_ch, out_channels=6, kernel_size=3, stride=1, padding=0, bias=True)
        # Max-pooling 1
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1, padding=0, bias=True)
        # Max-pooling 2
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Convolution 3
        self.conv3 = torch.nn.Conv2d(in_channels= 16, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
        # Max-pooling 3
        self.max_pool_3 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        self.fc1 = torch.nn.Linear(32 * 2 * 2, 120)     # convert matrix with 16*5*5 (= 400)  features to a matrix of 120 features (columns)
        self.fc2 = torch.nn.Linear(120, 84)             # convert matrix with 120 features to a matrix of 84 features (columns)
        self.fc3 = torch.nn.Linear(84, 10)              # convert matrix with 84 features to a matrix of 10 features (columns)
        self.fc4 = torch.nn.Linear(10, num_classes)     # convert matrix with 10 features to a matrix of 2 features (columns)

    def forward(self, x):
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv1(x))    #    6*28*28 (6*30*30, 6*32*32 w pad)
        # max-pooling with 2x2 grid
        x = self.max_pool_1(x)      # the size will be      6*14*14  (6*15*15, 6*16*16 w pad)
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv2(x))    #    16*10*10 (16*13*13, 16*16*16 w pad)
        # max-pooling with 2x2 grid
        x = self.max_pool_2(x)      # the size will be      16*5*5  (16*6*6, 16*8*8)
        # convolve, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.conv3(x))     #   32*1*1 (32*4*4, 32*8*8 w pad)
        # max-pooling with 1x1 grid
        x = self.max_pool_3(x)      # the size will be      32*0*0  (32*2*2, 32*4*4)

        # first flatten 'max_pool_2_out' to contain 16*5*5 columns
        # read through https://stackoverflow.com/a/42482819/7551231
        #x = x.view(-1, 32 * 2 * 2) # x = x.view(x.size(0), -1) this is same because x.size(0) is 1
        x = x.view(x.size(0), -1) # this layer (reshape is not supported in caffe)
        # FC-1, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc1(x))
        # FC-2, then perform ReLU non-linearity
        x = torch.nn.functional.relu(self.fc2(x))
        # FC-3, then perform ReLu non-linearity
        x = torch.nn.functional.relu(self.fc3(x))
        # FC-4
        x = self.fc4(x)

        return x


def test():
    in_ch = 3;
    num_classes = 2;
    net = LeNet5(num_classes, in_ch) # 2 classes with 64x64 size
    x = torch.randn(1,in_ch,32,32)
    y = net(x)
    print(y.size())

#test()
