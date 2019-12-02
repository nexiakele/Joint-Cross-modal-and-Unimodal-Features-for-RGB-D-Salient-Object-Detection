# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:47:56 2018

@author: Dell
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from layers import split_tensor
class edge_loss(nn.Module):
    def __init__(self):
        super(edge_loss, self).__init__() 
        self.a = torch.Tensor([[0, 0, 0],
                               [1.0, 0, -1.0],
                               [0, 0, 0]])
        self.a = self.a.view((1,1,3,3)).cuda()
        self.b = torch.Tensor([[0, 1.0, 0],
                                [0, 0, 0],
                                [0, -1.0,0]])
        self.b = self.b.view((1,1,3,3)).cuda()
        self.loss  = nn.SmoothL1Loss()
    def forward(self, logits, targets):
          with torch.no_grad():
                logits_x = F.conv2d(logits, self.a)
                logits_y = F.conv2d(logits, self.b)
                logits_G = torch.sqrt(torch.pow(logits_x,2)+ torch.pow(logits_y,2))
                targets_x = F.conv2d(targets, self.a)
                targets_y = F.conv2d(targets, self.b)
                targets_G = torch.sqrt(torch.pow(targets_x,2)+ torch.pow(targets_y,2))
          loss = self.loss(logits_G, targets_G)
          return loss
            
class fuse_loss(nn.Module):
    def __init__(self):
        super(fuse_loss, self).__init__() 
        self.bceloss = nn.BCELoss()
        self.edge_loss  = edge_loss()
    def forward(self, x, y):
          loss1 = self.bceloss(x,y)
          loss2 = self.edge_loss(x, y)
          return loss1 + loss2

def weighted_binary_cross_entropy_with_logits(logits, targets, loss_type = 2,
                                              weight=None, size_average=True, reduce=True):
    if not (targets.size() == logits.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), logits.size()))
    pos_weight = 1.0
    if   loss_type == 0 :
          n_p = targets.sum()
          n_n = (1- targets).sum()
          pos_weight = n_n / n_p
    elif loss_type == 1 :
          n_p = targets.sum()
          n_n = (1- targets).sum()
          pos_weight = n_n / n_p
          tp = (targets * logits).sum()
          tn = logits.sum() - tp
          t_weight = n_n*(n_p-tp) / (n_p*(n_n-tn))
          pos_weight =  pos_weight + t_weight
    elif loss_type == 2 :
          n_p = targets.sum()
          n_n = (1- targets).sum()
          pos_weight = n_n / n_p
          tp = (targets * logits).sum()
          tn = logits.sum() - tp
          t_weight = n_n*(n_p-tp) / (n_p*(n_n-tn))
          x1 = n_n / (n_p + n_n) + (n_p - tp) / n_p
          x2 = n_p / (n_p + n_n) + (n_n - tn)/n_n
          pos_weight =  x1/x2
    elif loss_type == 3 :
          n_p = targets.sum()
          n_n = (1- targets).sum()
          pos_weight = n_n / n_p
          tp = (targets * logits).sum()
          tn = logits.sum() - tp
          t_weight = n_n*(n_p-tp) / (n_p*(n_n-tn))
          pos_weight =  (pos_weight + t_weight) / 2.0
    
###############################################################################
    max_val = (-logits).clamp(min=0)
    log_weight = 1 + (pos_weight - 1) * targets
    loss = (1 - targets) * logits + log_weight * (((-max_val).exp() + (-logits - max_val).exp()).log() + max_val)
    return loss.mean()

class WeightedBCELoss(_Loss):
    def __init__(self, pos_weight=None, weight=None,
                 size_average=True, reduce=True,loss_type = 0):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()
        self.register_buffer('weight', weight)
        self.size_average = size_average
        self.reduce = reduce
        self.loss_type = loss_type
    def forward(self, input, target):
        if self.weight is not None:
            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                 loss_type = self.loss_type,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy_with_logits(input, target,
                                                 loss_type = self.loss_type,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
class Aux_weight_loss5(nn.Module):
    def __init__(self, weight = [1,1,1], loss_type = 0):
        super(Aux_weight_loss5, self).__init__() 
        self.main_loss = WeightedBCELoss(loss_type = loss_type)
        self.aux_loss1 = WeightedBCELoss(loss_type = loss_type)
        self.aux_loss2 = WeightedBCELoss(loss_type = loss_type)
        self.aux_loss3 = WeightedBCELoss(loss_type = loss_type)
        self.aux_loss4 = WeightedBCELoss(loss_type = loss_type)
        self.weight =weight
    def forward(self, out, gt, aux, aux_gt):
        main_loss = self.main_loss(out, gt)  
        au_loss1  = self.aux_loss1(aux[0], aux_gt[0]) * self.weight[0]
        au_loss2  = self.aux_loss2(aux[1], aux_gt[1]) * self.weight[1]
        au_loss3  = self.aux_loss3(aux[2], aux_gt[2]) * self.weight[2]
        au_loss4  = self.aux_loss4(aux[3], aux_gt[3]) * self.weight[3]
        record= [main_loss.item(), au_loss1.item(),au_loss2.item(), 
                 au_loss3.item(), au_loss4.item()]
        totall_loss = main_loss + au_loss1 + au_loss2 + au_loss3 + au_loss4
        return  totall_loss, record



class totall_loss_base(nn.Module):
    def __init__(self, loss_, w=1):
          super(totall_loss_base, self).__init__() 
          self.loss = loss_()
          self.w = w
    def forward(self, ts, gt):
        lrds = self.loss(ts[0], gt)
        lrs  = self.loss(ts[1], gt) * self.w
        lds = self.loss(ts[2], gt) * self.w
        totall = lrds+lrs+lds
        return  totall
###########################################
class head_Aux_fuse_loss_for_14_4(nn.Module):
    def __init__(self, weight):
        super(head_Aux_fuse_loss_for_14_4, self).__init__() 
        self.loss1  = fuse_loss()
        self.loss2  = nn.BCELoss()
        self.split = split_tensor(3)
    def forward(self,out, gt):
        fp = out[0]
        fpls = self.loss1(fp, gt) * 2.0
        flrd  = self.loss1(out[1], gt)
        flr  = self.loss1(out[2], gt)
        fld  = self.loss1(out[3], gt)
        mask1, mask2, mask3 = self.split(out[4])
        if flrd <= flr and flrd <= fld:
              sw = mask1
              if flr <= fld:
                    sw1 = mask2
              else:
                    sw1 = mask3
        elif flr < flrd and flr < fld:
              sw = mask2
              if flrd <= fld:
                    sw1 = mask1
              else:
                    sw1 = mask3
        else:
              sw = mask3
              if flrd <= flr:
                    sw1 = mask1
              else:
                    sw1 = mask2
        with torch.no_grad():
              psw = sw*gt + (1.0-sw)*(1.0-gt)
              psw = psw.clamp(min=0, max=1)
              psw1 = (1 - psw) * (sw1*gt + (1.0-sw1)*(1.0-gt))
              psw1 = psw1.clamp(min=0, max=1)
        sw_loss = self.loss2(sw, psw)*0.2
        sw_loss1 = self.loss2(sw1, psw1)*0.2
        loss = fpls + flrd + flr + fld + sw_loss+sw_loss1
        return  loss


class head_Aux_fuse_loss_for_14_5(nn.Module):
    def __init__(self, weight):
        super(head_Aux_fuse_loss_for_14_5, self).__init__() 
        self.loss1  = fuse_loss()
        self.loss2  = nn.BCELoss()
        self.split = split_tensor(3)
    def forward(self,out, gt):
        fp = out[0]
        fpls = self.loss1(fp, gt) * 2.0
        flrd  = self.loss1(out[1], gt)
        flr  = self.loss1(out[2], gt)
        fld  = self.loss1(out[3], gt)
        mask1, mask2, mask3 = self.split(out[4])
        with torch.no_grad():
              psw = out[1]*gt + (1.0-out[1])*(1.0-gt)
              psw = psw.clamp(min=0, max=1)
              psw1 = (1 - psw) * (out[2]*gt + (1.0-out[2])*(1.0-gt))
              psw1 = psw1.clamp(min=0, max=1)
        sw_loss = self.loss2(mask1, psw)*0.1
        sw_loss1 = self.loss2(mask2, psw1)*0.1
        loss = fpls + flrd + flr + fld + sw_loss+sw_loss1
        return  loss
def find_max(x1, x2, x3):
    if x1>x2 and x1 > x3:
        return 0
    elif x2 > x1 and x2 > x3:
        return 1
    else:
        return 2
class head_Aux_fuse_loss_for_14_6(nn.Module):
    def __init__(self, weight=1):
        super(head_Aux_fuse_loss_for_14_6, self).__init__() 
        self.loss1  = fuse_loss()
        self.loss2  = nn.BCELoss()
        self.loss3  = nn.L1Loss()
        self.w_loss = nn.CrossEntropyLoss()
        self.split = split_tensor(3)
    def forward(self,out, gt):
        fp = out[0]
        fpls = self.loss1(fp, gt) * 2.0
        flrd  = self.loss1(out[1], gt)
        flr  = self.loss1(out[2], gt)
        fld  = self.loss1(out[3], gt)
        out_rd, out_r, out_d, out_weight = out[1], out[2], out[3], out[4]
        
        weight_gt = []
        for o_rd, o_r, o_d, g in zip(out_rd, out_r, out_d, gt):
            with torch.no_grad():
                l_rd = self.loss2(o_rd, g) + self.loss3(o_rd, g)
                l_r  = self.loss2(o_r , g) + self.loss3(o_r , g)
                l_d  = self.loss2(o_d,  g) + self.loss3(o_d, g)
            weight_gt.append(find_max(l_rd, l_r, l_d))
        weight_gt = torch.tensor(weight_gt).cuda()  
        sw_loss = self.w_loss(out_weight.squeeze(), weight_gt)
        loss = fpls + flrd + flr + fld + sw_loss
        return  loss
class Aux_fuse_loss_for_14_2(nn.Module):
    def __init__(self, weight = [0.2,0.4,0.6,0.8,1], head = head_Aux_fuse_loss_for_14_4):
        super(Aux_fuse_loss_for_14_2, self).__init__() 
        self.aux_loss1 =   head(weight[0])
        self.aux_loss2 =   totall_loss_base(fuse_loss, weight[1])
        self.aux_loss3 =   totall_loss_base(fuse_loss, weight[2])
        self.aux_loss4 =   totall_loss_base(nn.BCELoss, weight[3])
        self.aux_loss5 =   totall_loss_base(nn.BCELoss, weight[4])
    def forward(self,out, gt):
        au_loss1  = self.aux_loss1(out[0], gt[0])
        au_loss2  = self.aux_loss2(out[1], gt[1])
        au_loss3  = self.aux_loss3(out[2], gt[2])
        au_loss4  = self.aux_loss4(out[3], gt[3])
        au_loss5  = self.aux_loss5(out[4], gt[4])
        record= [au_loss1.item(),au_loss2.item(),au_loss3.item(),au_loss4.item(),au_loss5.item()]
        totall_loss = au_loss1+au_loss2+au_loss3+au_loss4+au_loss5
        return  totall_loss, record

###########################################
class head_Aux_fuse_loss_for_15(nn.Module):
    def __init__(self, weight):
        super(head_Aux_fuse_loss_for_15, self).__init__() 
        self.fpl  = fuse_loss()
    def forward(self,out, gt):
        fpls = self.fpl(out[0], gt) * 2.0
        frd = self.fpl(out[1], gt)
        frr = self.fpl(out[2], gt)
        fdd = self.fpl(out[3], gt)
        return fpls+frd+frr+fdd
class Aux_fuse_loss_for_15(nn.Module):
    def __init__(self, weight = [0.2,0.4,0.6,0.8,1]):
        super(Aux_fuse_loss_for_15, self).__init__() 
        self.aux_loss1 =   head_Aux_fuse_loss_for_15(weight[0])
        self.aux_loss2 =   totall_loss_base(fuse_loss, weight[1])
        self.aux_loss3 =   totall_loss_base(fuse_loss, weight[2])
        self.aux_loss4 =   totall_loss_base(nn.BCELoss, weight[3])
        self.aux_loss5 =   totall_loss_base(nn.BCELoss, weight[4])
    def forward(self,out, gt):
        au_loss1  = self.aux_loss1(out[0], gt[0])
        au_loss2  = self.aux_loss2(out[1], gt[1])
        au_loss3  = self.aux_loss3(out[2], gt[2])
        au_loss4  = self.aux_loss4(out[3], gt[3])
        au_loss5  = self.aux_loss5(out[4], gt[4])
        record= [au_loss1.item(),au_loss2.item(),au_loss3.item(),au_loss4.item(),au_loss5.item()]
        totall_loss = au_loss1+au_loss2+au_loss3+au_loss4+au_loss5
        return  totall_loss, record
    
class loss5(nn.Module):
    def __init__(self):
        super(loss5, self).__init__() 
        self.main_loss =   fuse_loss()
        self.aux_loss   =   fuse_loss()
        self.aux_loss2  =   nn.BCELoss()
    def forward(self,out, gt):
        main_loss = self.main_loss(out[0], gt[0]) * 2.0
        au1  = self.aux_loss(out[1], gt[1])
        au2  = self.aux_loss(out[2], gt[2])
        au3  = self.aux_loss2(out[3], gt[3])
        au4  = self.aux_loss2(out[4], gt[4])
        record= [main_loss.item(),au1.item(),au2.item(),au3.item(),au4.item()]
        totall_loss = main_loss+au1+au2+au3+au4
        return  totall_loss, record    
##############################################################################
def get_loss(loss_type = 0, weight=[1,1,1,1,1]):
      if loss_type == 0:
            loss = Aux_weight_loss5(weight)
      elif loss_type == 15:
            loss = Aux_fuse_loss_for_15([1,1,1,1,1])
      elif loss_type == 22:
            loss = Aux_fuse_loss_for_14_2([1,1,1,1,1],head_Aux_fuse_loss_for_14_4)
      elif loss_type == 23:
            loss = Aux_fuse_loss_for_14_2([1,1,1,1,1],head_Aux_fuse_loss_for_14_5)
      elif loss_type == 24:
            loss = Aux_fuse_loss_for_14_2([1,1,1,1,1],head_Aux_fuse_loss_for_14_6)
      elif loss_type == 1:  
            loss = loss5()  
      return loss