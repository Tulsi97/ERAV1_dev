import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import torch
from torch import nn
from torchinfo import summary
from model import MyResNet, ResBlock

def get_lr(optimizer):
    """
    for tracking how your learning rate is changing throughout training
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device

def print_summary(model, device, input_size=(3,32,32)):
    device = get_device()
    model1 = MyResNet().to(device)
    summary(model, device, input_size)
    
