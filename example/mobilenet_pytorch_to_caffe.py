import sys

sys.path.insert(0, '.')
import torch
from torch.autograd import Variable
from torchvision.models import resnet
import pytorch_to_caffe
from MobileNet import MobileNet

"""
def load_model(input_size, device='cpu'):
    # Model
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('==> Building model..')
    net = MobileNet(2, input_size)
    net = net.to(device)

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('./checkpoint/ckpt_66_99.648587.pth')
    checkpoint = torch.load('./checkpoint/ckpt_318_99.730778.pth')
    net.load_state_dict(checkpoint['weight'])
    return net
    
"""

if __name__ == '__main__':
    input_size =64
    number_classes =2
    name = 'MobileNet'

    #hardware setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu' # now it works only under 'cpu' settings

    net = MobileNet(number_classes, input_size)

    #if(device == 'cuda'):
    net = net.to(device)

    # load a pre-trained pyTorch model
    checkpoint = torch.load("./ckpt_123_99.970.pth")
    net.load_state_dict(checkpoint['weight'])

    # if u want to use cpu, then you need to do something
    # net = net.to('cpu')
    # input_ = input_.to('cpu')

    net.eval()

    input_ = torch.ones([1, 3, input_size, input_size])
    input = input_.to(device)
    # input=torch.ones([1,3,224,224])
    # cuda problem ...
    pytorch_to_caffe.trans_net(net, input, name)

    pytorch_to_caffe.save_prototxt('{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('{}.caffemodel'.format(name))