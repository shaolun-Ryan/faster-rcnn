import torch
import os

try:
    torch.cuda.is_available()
    print('Remote server is connected, and cuda is available.')
except:
    raise ValueError('Cuda is not available on, please check GPU status.')
