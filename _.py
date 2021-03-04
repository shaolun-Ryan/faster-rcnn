import torch
import os

aa = 123
bb = 456
c = 'shaolun'

list = [1, 2, 3, 4, 5]

for item in list:
    print(item)

cc = bb+aa if aa == 123 else bb-aa

assert isinstance(c, str)
# print(type(aa), type(c))

try:
    a = torch.cuda.is_available()
    print('Remote server is connected, and cuda is available.')
except:
    raise ValueError('Cuda is not available on, please check GPU status.')
