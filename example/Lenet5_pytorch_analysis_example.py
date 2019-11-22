#coding=utf-8
import torch
import pytorch_analyser
from LeNet5 import LeNet5

if __name__=='__main__':
    # customized net :LeNet5.
    input_size = 32 # noew lenet is 32x32 only
    number_classes = 2
    in_channel = 3;
    name = 'LeNet5{}x{}'.format(input_size, input_size)

    # hardware setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'  # now it works only under 'cpu' settings
    net = LeNet5(number_classes, in_channel)
    # if(device == 'cuda'):
    net = net.to(device)
    # load a pre-trained pyTorch model
    checkpoint = torch.load("./lenet32x32_ckpt_10_60.667504.pth")
    net.load_state_dict(checkpoint['weight'])


    input_ = torch.ones([1, in_channel, input_size, input_size])
    input = input_.to(device)

    blob_dict, tracked_layers = pytorch_analyser.analyse(net, input)
    pytorch_analyser.save_csv(tracked_layers, './tmp/'+name+'_analysis.csv')