import torch.nn as nn
import torch
import numpy as np

class PAM_Module(nn.Module):
    def __init__(self,in_dim):
        super(PAM_Module,self).__init__()
        self.chanel_in=in_dim
        self.qc=nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.kc=nn.Conv2d(in_channels=in_dim,out_channels=in_dim//8,kernel_size=1)
        self.vc=nn.Conv2d(in_channels=in_dim,out_channels=in_dim,kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        B, C, width, height = x.size()
        img_q=self.qc(x).view(B,-1,width*height).permute(0,2,1)
        img_k=self.kc(x).view(B,-1,width*height)
        energy=torch.bmm(img_q,img_k)
        attention=self.softmax(energy)

        img_v=self.vc(x).view(B,-1,width*height)
        out=torch.bmm(img_v,attention.permute(0,2,1))
        out=out.view(B,C,width,height)
        out=self.gamma*out+x
        return out
class CAM_Module(nn.Module):
    def __init__(self,in_dim):
        super(CAM_Module,self).__init__()
        self.chanel_in=in_dim

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        B, C, width, height = x.size()
        img_q=x.view(B,C,-1)
        img_k=x.view(B,C,-1).permute(0,2,1)
        energy=torch.bmm(img_q,img_k)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention=self.softmax(energy_new)

        img_v=x.view(B,C,-1)
        out=torch.bmm(attention,img_v)
        out=out.view(B,C,width,height)
        out=self.gamma*out+x
        return out