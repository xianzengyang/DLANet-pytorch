import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
from Utils.GCL import GatedSpatialConv2d
import Utils.Resnet
from Utils.Attention import PCCAModule

class conv_block(nn.Module):
    """
    Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x
class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
class DLANet(nn.Module):
    def __init__(self,in_ch=3, out_ch=1):

        super(DLANet,self).__init__()
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.PCCA = PCCAModule(filters[4], filters[3])

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=filters[3], out_channels=filters[3], kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=False),
            nn.BatchNorm2d(filters[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters[3], out_channels=filters[2], kernel_size=1, stride=1, padding=0, dilation=1,
                      bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True)
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=152, out_channels=filters[2], kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters[2], out_channels=filters[2], kernel_size=3, stride=1, padding=1, dilation=1,
                      bias=False),
            nn.BatchNorm2d(filters[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters[2], out_channels=out_ch, kernel_size=1, stride=1, padding=0, dilation=1,
                      bias=False)
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels=filters[2], out_channels=24, kernel_size=1, stride=1, padding=0, dilation=1,
                      bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )
        self.active = torch.nn.Sigmoid()
        ##################################################################################
        self.dsn2 = nn.Conv2d(filters[1], 1, 1)
        self.dsn3 = nn.Conv2d(filters[2], 1, 1)
        self.dsn4 = nn.Conv2d(filters[3], 1, 1)
        self.dsn5 = nn.Conv2d(filters[4], 1, 1)
        #
        self.gate1=GatedSpatialConv2d(32,32)
        self.gate2=GatedSpatialConv2d(16,16)
        self.gate3=GatedSpatialConv2d(8,8)
        self.gate4=GatedSpatialConv2d(4,4)
        #
        self.d1 = nn.Conv2d(32, 16, 1)
        self.d2 = nn.Conv2d(16, 8, 1)
        self.d3 = nn.Conv2d(8, 4, 1)
        #
        self.res1 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.res2 = Resnet.BasicBlock(32, 32, stride=1, downsample=None)
        self.res3 = Resnet.BasicBlock(16, 16, stride=1, downsample=None)
        self.res4 = Resnet.BasicBlock(8, 8, stride=1, downsample=None)
        #
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)
        #############################################################################
        self.fusionP1=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusionP2=nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusionC1 = nn.Sequential(nn.Conv2d(2,32,3,1,1),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(inplace=True)
                                    )
        self.fusionC2 = nn.Sequential(nn.Conv2d(32,64,3,1,1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True)
                                      )
        self.fusionC3 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,stride=1,dilation=2,padding=2),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(128,128,kernel_size=3,stride=1,dilation=2,padding=2),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))
        self.fusionUP1 = up_conv(128,128)
        self.fusionUc1 = conv_block(128,64)
        self.fusionUP2 = up_conv(64,64)
        self.fusionUc2 = conv_block(64,32)
        self.fusionCon = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1))
        self._initialize_weights()
    def forward(self,x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e3_ = self.reduce(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        eout = self.PCCA(e5)

        d5 = self.fc1(eout)

        d3 = F.interpolate(d5, size=e3_.shape[2:], mode="bilinear", align_corners=False)
        d3 = torch.cat((d3, e3_), dim=1)
        d3 = self.fc2(d3)
        d1 = F.interpolate(d3, size=x.shape[2:], mode="bilinear", align_corners=False)

        outs= self.active(d1)
        ###############################
        s2 = F.interpolate(self.dsn2(e2),  size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.dsn3(e3),  size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.dsn4(e4),  size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        s5 = F.interpolate(self.dsn5(e5), size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        s1 = F.interpolate(e1,size=x.shape[2:], mode='bilinear', align_corners=True)
        b1 = self.res1(s1)
        b1 = F.interpolate(b1, size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        be1=self.gate1(b1,s2)

        b2=self.res2(be1)
        b2 = F.interpolate(b2, size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        b2=self.d1(b2)
        be2=self.gate2(b2,s3)

        b3=self.res3(be2)
        b3= F.interpolate(b3, size=x.shape[2:],
                           mode='bilinear', align_corners=True)
        b3=self.d2(b3)
        be3=self.gate3(b3,s4)

        b4=self.res4(be3)
        b4= F.interpolate(b4,size=x.shape[2:],
                          mode='bilinear',align_corners=True)
        b4=self.d3(b4)
        be4 = self.gate4(b4,s5)

        be4=self.fuse(be4)

        fusee = F.interpolate(be4, size=x.shape[2:],
                             mode='bilinear', align_corners=True)
        oute=self.active(fusee)
        ###############################################
        fudata=torch.cat((outs,oute),dim=1)
        fu1=self.fusionC1(fudata)
        fu2=self.fusionP1(fu1)
        fu3=self.fusionC2(fu2)
        fu4=self.fusionP2(fu3)
        fu5=self.fusionC3(fu4)
        fu6=self.fusionUP1(fu5)
        fu7=self.fusionUc1(fu6)
        fu8=self.fusionUP2(fu7)
        fu9=self.fusionUc2(fu8)
        fusion=self.fusionCon(fu9)
        fusion=self.active(fusion)
        ###############################################
        results = [outs,oute,fusion]
        return results
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

if __name__ == "__main__":
    model = DLANet()
    model.eval()
    image = torch.randn(1, 3, 400, 400)

    print(model)
    print("input:", image.shape)
    print("output:", model(image)[0].shape)