#coding=utf-8
import torch
import pytorch_analyser
from MobileNet import MobileNet

if __name__=='__main__':
    # customized net :mobilenet v1.
    input_size = 64
    number_classes = 2
    name = 'MobileNet{}x{}'.format(input_size, input_size)

    # hardware setting
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'  # now it works only under 'cpu' settings
    net = MobileNet(number_classes, input_size)
    # if(device == 'cuda'):
    net = net.to(device)
    # load a pre-trained pyTorch model
    checkpoint = torch.load("./best_ckpt_64x64_20190829.pth")
    net.load_state_dict(checkpoint['weight'])


    input_ = torch.ones([1, 3, input_size, input_size])
    input = input_.to(device)

    blob_dict, tracked_layers = pytorch_analyser.analyse(net, input)
    pytorch_analyser.save_csv(tracked_layers, './tmp/'+name+'_analysis.csv')