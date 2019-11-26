import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        loss = sigmoid_focal_loss_cpu(logits, targets, self.gamma, self.alpha)
        pos_inds = torch.nonzero(targets > 0).squeeze(1)
        N = targets.size(0)
        loss = loss.sum() / (pos_inds.numel() + N)
        return loss

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr

class SoftDiceLoss(nn.Module):
    def __init__(self, beta):
        super(SoftDiceLoss, self).__init__()
        self.beta = beta
    
    def forward(self, logits, targets):
        smooth = 1.
        intersection = logits * targets
        false_negative = (1-logits) * targets
        score = (intersection.sum(1) + self.beta**2 * intersection.sum(1) + smooth) / (logits.sum(1) + targets.sum(1) + (self.beta**2 - 1) * intersection.sum(1) + (self.beta**2 - 1) * false_negative.sum(1) + smooth)
        dice = score.sum() / targets.size(0)
        return 1. - dice

class FocalDiceLoss(nn.Module):
    def __init__(self):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = SigmoidFocalLoss()
        self.dice_loss = SoftDiceLoss()

    def forward(self, logits, targets):
        focal_loss = self.focal_loss(logits, targets)
        dice_loss = 0.1*self.dice_loss(torch.sigmoid(logits), targets)
        loss = focal_loss + dice_loss
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, fw=1, dw=0.1, bw=0.1, beta=1):
        super(CombinedLoss, self).__init__()
        self.focal_loss = SigmoidFocalLoss()
        self.dice_loss = SoftDiceLoss(beta)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.fw = fw
        self.dw = dw
        self.bw = bw

    def forward(self, logits, targets):
        focal_loss = self.fw*self.focal_loss(logits, targets)
        dice_loss = self.dw*self.dice_loss(torch.sigmoid(logits), targets)
        bce_loss = self.bw*self.bce_loss(logits, targets)
        loss = focal_loss + dice_loss + bce_loss
        return loss

def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    p = torch.sigmoid(logits)
    term1 = (1 - p) ** gamma * torch.log(p)
    term2 = p ** gamma * torch.log(1 - p)
    return -targets * term1 * alpha - (1 - targets) * term2 * (1 - alpha)
               
