import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from DLANet.DLANet import DLANet
import pandas as pd
import argparse
from function.MyDataLoader import Testread
from PIL import Image
import os, sys
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
import re

parser=argparse.ArgumentParser(description="Mynet.train")
parser.add_argument("--batchsize",type=int,default=1)
parser.add_argument("--test_path",type=str,default=r".\date\test\image")
parser.add_argument("--model_path",type=str,default=r".\modelsave\ew0.pth")
parser.add_argument("--save_path",type=str,default=r".\result")
parser.add_argument("--nl_path",type=str,default=r".\test.lst")
args=parser.parse_args()

def parasum(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

def nameget(list):
    with open(list, 'r') as f:
        filelist = f.readlines()
        for i in range(len(filelist)):
             filelist[i]=filelist[i].split('\n')[0]
    return(filelist)

def test(models,test_dir,save_dir,namel):
    device = torch.device("cuda")
    model = Senet()
    model.cuda()
    model.load_state_dict(torch.load(models))
    parasum(model)
    model.to(device)
    model.eval()
    testset=Testread(test_dir)
    test_loader=torch.utils.data.DataLoader(testset, batch_size=args.batchsize,drop_last=True,shuffle=False)

    for idx, image in enumerate(test_loader):
        image=image.to(device)
        result=model(image)
        result[2] = torch.squeeze(result[2])
        result[2][result[2] >= 0.5] = 1
        result[2][result[2] < 0.5] = 0
        result[2] = result[2].detach().cpu().numpy()
        result3 = Image.fromarray((result[2] * 255).astype(np.uint8))
        result3.save(join(save_dir, "%s" % namel[idx]))

if __name__ == '__main__':
    torch.cuda.set_device(0)
    name = nameget(args.nl_path)
    # name = re.findall(r"\d*\d", str(name))
    if os.path.exists(args.save_path)!=True:
        os.makedirs(args.save_path)
    test(args.model_path,args.test_path,args.save_path,name)