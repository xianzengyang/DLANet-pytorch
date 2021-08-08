import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter
import argparse
from DLANet.DLANet import DLANet
import pandas as pd
from function.MyDataLoader import Dataseter
from function.loss import focal_loss,Edge_loss
####################################################
#para
####################################################
parser=argparse.ArgumentParser(description="Mynet")
parser.add_argument("--BatchSize",type=int,default=4)
parser.add_argument("--lr",type=float,default=0.001)
parser.add_argument("--MaxEpoch",type=int,default=100)
parser.add_argument("--TLoss",type=float,default=0)
parser.add_argument("--VLoss",type=float,default=0)
parser.add_argument("--GPU",type=str,default="cuda:0")
parser.add_argument("--MaxLoss",type=float,default=10000.)
parser.add_argument("--trainimage",type=str,default=r".\date\train\image3")
parser.add_argument("--trainlabel",type=str,default=r".\date\train\polyLabel3")
parser.add_argument("--trainline",type=str,default=r".\date\train\lineLabel3")
parser.add_argument("--vaildimage",type=str,default=r".\date\vaild\image")
parser.add_argument("--vaildlabel",type=str,default=r".\date\vaild\label")
parser.add_argument("--vaildline",type=str,default=r".\date\vaild\labeledge")
parser.add_argument("--weight_decay",type=float,default=1e-5)
args=parser.parse_args()
####################################################
#datasat
####################################################
train_image=args.trainimage
labelpoly_image=args.trainlabel
labelline_image=args.trainline
valid_image=args.vaildimage
valid_labelpoly=args.vaildlabel
valid_labelline=args.vaildline

Training_Data = Dataseter(train_image,labelpoly_image,labelline_image)
Validing_Data = Dataseter(valid_image,valid_labelpoly,valid_labelline)

train_loader = torch.utils.data.DataLoader(Training_Data, batch_size=args.BatchSize, drop_last=True,shuffle=True)
vaild_loader = torch.utils.data.DataLoader(Validing_Data, batch_size=args.BatchSize, drop_last=True,shuffle=True)
####################################################
#model & loss & opt
####################################################
model=Senet()
#model.load_state_dict(torch.load(r'E:\小论文\论文代码\TestNetProject4(FuisionNet)\Model\Fusionw3.pth'))
optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
####################################################
#train
####################################################
device=torch.device(args.GPU)
model.to(device)
LossL={}
VLossL={}
for Epoch in range(args.MaxEpoch):
    Loss=args.TLoss
    VLoss=args.VLoss
    for batch_idx, data in enumerate(train_loader):
        sideloss=0
        image, labelpoly,labelline = data
        image, labelpoly,labelline = image.to(device),labelpoly.to(device),labelline.to(device)
        optimizer.zero_grad()

        image=image.type(torch.FloatTensor)
        image=image.to(device)
        outputs=model(image)

        tedge_loss=Edge_loss()
        tdice_loss=dice_loss()
        # for o in range(2,7):
        #     sideloss=0.1*tedge_loss(outputs[o],labelline)

        loss=tdice_loss(outputs[0],labelpoly)+tedge_loss(outputs[1],labelline)+tdice_loss(outputs[2],labelpoly)
        loss.requires_grad_(True)
        loss.backward()

        optimizer.step()
        Loss = Loss+loss.item()

    with torch.no_grad():
        for batch_idx, data in enumerate(vaild_loader):
            image, labelpoly, labelline = data
            image, labelpoly, labelline = image.to(device), labelpoly.to(device), labelline.to(device)

            image = image.type(torch.FloatTensor)
            image = image.to(device)
            outputs = model(image)

            vedge_loss = Edge_loss()
            vdice_loss = dice_loss()

            # for o in range(2, 7):
            #     vsideloss = 0.1 * vedge_loss (outputs[o], labelline)

            vloss = vdice_loss(outputs[0], labelpoly) + vedge_loss(outputs[1],labelline) + vdice_loss(outputs[2], labelpoly)
            vloss.requires_grad_(True)
            vloss.backward()

            VLoss = VLoss + vloss.item()
    print('[epoch: %d] loss:%.6f lr:%.9f valid_loss:%.6f' % (Epoch + 1, Loss / len(train_loader), optimizer.state_dict()['param_groups'][0]['lr'],VLoss / len(vaild_loader)))
    scheduler.step(VLoss/len(vaild_loader))

    LossL[Epoch]=Loss/(len(train_loader))
    VLossL[Epoch]=VLoss/(len(vaild_loader))

    if VLoss<args.MaxLoss:
        torch.save(model.state_dict(), f=r'.\modelsave\epoch{}.pth'.format(Epoch+1))
        args.MaxLoss=VLoss
######################################################
##########loss_line_save#########################################
Loss_N=list(LossL.values())
VLoss_N=list(VLossL.values())
dataTloss=pd.DataFrame(Loss_N)
dataVloss=pd.DataFrame(VLoss_N)
writer = pd.ExcelWriter('LossLine.xlsx')
dataTloss.to_excel(writer, 'page_1', float_format='%.5f')
dataVloss.to_excel(writer, 'page_2', float_format='%.5f')
writer.save()
writer.close()

