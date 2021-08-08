import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class focal_Loss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.75, reduction='mean'):
        super(focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def forward(self,prediction,target):
        smooth = 1.0

        i_flat = prediction.view(-1)
        t_flat = target.view(-1)

        intersection = (i_flat * t_flat).sum()
        return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
class Edge_loss(nn.Module):
    def __init__(self):
        super(Edge_loss, self).__init__()

    def forward(self,prediction, label):
        label = label.long()
        mask = label.float()
        num_positive = torch.sum((mask==1).float()).float()
        num_negative = torch.sum((mask==0).float()).float()

        mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
        mask[mask == 2] = 0
        cost = torch.nn.functional.binary_cross_entropy(
                prediction.float(),label.float(), weight=mask, reduction='none')
        return torch.mean(cost)