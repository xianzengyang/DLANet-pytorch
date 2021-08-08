import os
from PIL import Image
import torch
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
import random
import numpy as np
import cv2
import torchvision

class Dataseter(Dataset):
    def __init__(self,img_dir,lab1_dir,lab2_dir):
        super(Dataseter,self).__init__()
        self.images=sorted(os.listdir(img_dir))
        self.label1s=sorted(os.listdir(lab1_dir))
        self.label2s=sorted(os.listdir(lab2_dir))
        self.img_dir=img_dir
        self.lab1_dir=lab1_dir
        self.lab2_dir=lab2_dir

    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        #for i in range(len(self.images)):
        image=Image.open(self.img_dir +'/'+self.images[i])
        label1=Image.open(self.lab1_dir +'/'+self.label1s[i])
        label2=Image.open(self.lab2_dir+'/'+self.label2s[i])
        #print(self.images[i],self.labels[i])
        image = np.array(image, dtype=np.float32)
        label1 = np.array(label1, dtype=np.float32)
        label2 = np.array(label2, dtype=np.float32)

        image = np.transpose(image[:,:,0:3], (2, 0, 1))
        image=torch.from_numpy(image).float()
        label1=torch.from_numpy(label1).float().unsqueeze(0)
        label2 = torch.from_numpy(label2).float().unsqueeze(0)

        #image=image/255.0

        label1[label1>=0.5]=1
        label1[label1<0.5]=0
        label2[label2>=0.5]=1
        label2[label2<0.5]=0

        image = torch.Tensor(image)
        label1 = torch.Tensor(label1)
        label2= torch.Tensor(label2)
        return image,label1,label2

        #image=image.permute(2,0,1)
class Testread(Dataset):
    def __init__(self,testimg_dir):
        super(Testread,self).__init__()
        self.images = sorted(os.listdir(testimg_dir))
        self.testimg_dir=testimg_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = Image.open(self.testimg_dir + '/' + self.images[item])
        image = np.array(image, dtype=np.float32)
        image = np.transpose(image[:,:,0:3], (2, 0, 1))
        image = torch.from_numpy(image).float()

        #image = image / 255.0
        image = torch.Tensor(image)
        return image








