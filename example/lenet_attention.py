'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            #nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            #nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            #nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=False)


    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        #return x*psi
        print('x size:{}'.format( x.size()[1]))
        psi1 = psi.repeat(1,28,1,1)
        print('psi1 size:{}'.format(psi1.size()))
        #return psi.mul(x)
        return torch.mul(psi1, x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3)





    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))

        return out

class LeNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.layer1 = self._make_layer(in_planes, 50)   # 14 : 1/2 of last filter size 28 (in_planes, 14)

        self.layer2 = self._make_layer(50, 28)          # here fileter size max: 29 is max (14, 28)
        self.layer2_1 = self._make_layer(50, 28)
        self.layer3 = self._make_layer(48, 40)         # Ths layer can not be used due to the size of input (64 128)
        #self.Att1 = Attention_block(50, 50, 50)
        self.Att2 = Attention_block(28, 28, 28)
        #self.fc1   = nn.Linear(16*5*5, 120)
        #self.fc1 = nn.Linear(128*5*7, 256)
        self.fc1 = nn.Linear(28 * 5 * 7, 256)           # according to the above settings
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

        out_0 = self.layer2(out)
        out_1 = self.layer2_1(out)
        out = self.Att2(out_1, out_0)
        print(out.shape)
        out = F.max_pool2d(out, 2)
        #out = F.adaptive_max_pool2d(out, 2)

        #out = self.layer3(out)
        #out = F.max_pool2d(out, 2)
        

        # out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        return out
