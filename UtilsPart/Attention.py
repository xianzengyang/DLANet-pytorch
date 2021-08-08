import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,height,height).permute(0,2,1,3)
        # energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
        #                                                                                              height,
        #                                                                                              height).permute(0,
        #                                                                                                              2,
        #                                                                                                              1,
        #                                                                                                              3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

class PCCAModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PCCAModule, self).__init__()
        inter_channels = in_channels // 4
        self.conva = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   nn.BatchNorm2d(inter_channels))
        self.cca = CrissCrossAttention(inter_channels)
        self.convb = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(inter_channels))
        self.se=SE(inter_channels)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(3*inter_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
           nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x):
        output = self.conva(x)
        outputC1 = self.cca(output)
        outputC2 = self.cca(outputC1)
        outputC = self.convb(outputC2)

        outputS = self.se(output)
        outputS = self.convb(outputS)

        sum1=torch.cat([output, outputC], 1)
        sum2=torch.cat([sum1, outputS], 1)

        output = self.bottleneck(sum2)
        return output

class SE(nn.Module):
    def __init__(self,nin,reduce=16):
        super(SE,self).__init__()
        self.gp=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Linear(nin, nin // reduce),
            nn.ReLU(inplace=True),
            nn.Linear(nin // reduce, nin),
            nn.Sigmoid()
        )
    def forward(self,inputs):
        x = inputs
        b, c, _, _ = x.size()
        y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        return y

if __name__ == '__main__':
    model = RCCAModule(32,6)
    device = torch.device("cpu")

    model.to(device)
    x = torch.randn(2,32,400, 400)
    out = model(x)
    print(out.shape)