from thop import profile
from PCUNetS import PCNet
import torch

net=PCNet()
image = torch.randn(1, 3, 400, 400)
flops, params = profile(net, (image, ))
print(flops,params)