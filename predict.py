# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 20:55:18 2018

@author: Dell
"""
import torch
from torch import nn
import torch.nn.functional as F
import layers as _layers
###############################################################################
##########################predict##############################################
###############################################################################
class Predict3(nn.Module):
    def __init__(self, in_channel, out_channel = 1,k_s = 1, selection=None):
        super(Predict3, self).__init__() 
        
        self.conv2 = nn.Sequential( _layers.conv1x1(in_channel[1],out_channel),
                                    nn.Sigmoid())
        self.conv3 = nn.Sequential( _layers.conv1x1(in_channel[2],out_channel),
                                    nn.Sigmoid())
        self.conv4 = nn.Sequential( _layers.conv1x1(in_channel[0],out_channel),
                                    nn.Sigmoid())
        inc = in_channel[0] + in_channel[1] + in_channel[2]
        self.conv1 = nn.Sequential( _layers.conv1x1(inc,out_channel),
                                    nn.Sigmoid())
        
        self.select = selection(inc, inc)
    def forward(self, rd, r, d):
        srd = self.conv4(rd)
        srr = self.conv2(r)
        sdd = self.conv3(d)
        feature1 = torch.cat((rd, r, d), 1)
        weight = self.select(feature1)
        out = feature1 * weight.expand_as(feature1)
        out = self.conv1(out)
        return out,srd, srr, sdd
class convkx1x1xk3(nn.Module):
    def __init__(self, inc, outc,k_size=3, groups=1):
        super(convkx1x1xk3, self).__init__()
        self.convkx1xk = nn.Sequential(
                        nn.Conv2d(inc, outc,kernel_size=(k_size,1),
                                  padding=0,stride=1, groups=groups),
                        nn.Conv2d(inc, outc,kernel_size=(1, k_size),
                                  padding=0,stride=1, groups=groups),)
        self.conv1xkx1 = nn.Sequential(
                         nn.Conv2d(inc, outc,kernel_size=(1, k_size),
                                  padding=0,stride=1, groups=groups),
                         nn.Conv2d(inc, outc,kernel_size=(k_size,1),
                                  padding=0,stride=1, groups=groups),)
        self.Relu = nn.ReLU(inplace=True)                
    def forward(self, x):
        out1 = self.conv1xkx1(x)
        out2 = self.convkx1xk(x)
        out = torch.cat((out1, out2),1)
        out = self.Relu(out)
        return out    
class select(nn.Module):
    def __init__(self, inc, outc, reduction=4):
        super(select, self).__init__()
        self.m_c = inc // reduction
        self.outc = outc
        self.inc = inc
        pw = 28
        self.adapool = nn.AdaptiveAvgPool2d((pw,pw))
        self.inner_c = convkx1x1xk3(inc, inc, pw)
        self.fc = nn.Sequential(
            nn.Linear(inc*2, outc, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.adapool(x)
        y = self.inner_c(y).view(b, self.inc*2 )
        y = self.fc(y).view(b, self.outc, 1, 1)
        return y   

def get_predict(in_channel = 128, predict_type = 0):
      if predict_type == 1:
            predict = Predict3(in_channel, 1, selection=select)       
      
      return predict
  
if __name__ == '__main__':
    x=torch.rand(2,3)
    y = torch.zeros(2,3).long()
    ind= torch.argmax(x, dim=1)
    ind= ind.squeeze()
    print(ind)
    for i in range(2):
        y[i,ind[i]] = 1
    print(y)
