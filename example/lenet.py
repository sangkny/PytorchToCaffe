'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)





    def forward(self, x):
        out = F.relu(self.conv1(x))
        #out = F.relu(self.conv2(out))

        return out

class LeNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = self._make_layer(in_planes, 64)
        self.layer2 = self._make_layer( 64, 128)
        self.layer3 = self._make_layer(64, 128)
        #self.fc1   = nn.Linear(16*5*5, 120)
        self.fc1 = nn.Linear(128*6*8, 256)
        self.fc2   = nn.Linear(256, 84)
        self.fc3   = nn.Linear(84, out_planes)

    def _make_layer(self, inplane, outplane):

        layers = []
        layers.append(BasicBlock(inplane,outplane))

        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.conv1(x))
        out = self.layer1(x)
        out = F.max_pool2d(out, 2)
        #out = F.relu(self.conv2(out))
        out = self.layer2(out)
        out = F.max_pool2d(out, 2)

        #out = self.layer3(out)
        #out = F.max_pool2d(out, 2)

        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
