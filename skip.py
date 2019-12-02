# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:16:00 2018

@author: Dell
"""
import torch
from torch import nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import layers as _layer
#from torch_deform_conv.layers import ConvOffset2D
###############################################################################
###############################################################################
###############################################################################
class dilated_conv(nn.Module):
      def __init__(self, in_channel, out_channel, dilated = 1):
        super(dilated_conv, self).__init__()  
        self.dilated1 = nn.Sequential(
                                nn.Conv2d(in_channel, out_channel,kernel_size=3,stride=1,
                                          padding=dilated, dilation=dilated, bias=False),
                                nn.BatchNorm2d(out_channel),)
      def forward(self,x):
            out = self.dilated1(x)
            return out
class dilated_conv2(nn.Module):
      def __init__(self, in_channel, out_channel, dilated = 1):
        super(dilated_conv2, self).__init__()  
        self.dilated1 = nn.Sequential(
                                nn.Conv2d(in_channel, out_channel,kernel_size=3,stride=1,
                                          padding=dilated, dilation=dilated, bias=False),
                                nn.BatchNorm2d(out_channel),
                                nn.ReLU())
      def forward(self,x):
            out = self.dilated1(x)
            return out


                
class skip_basic(nn.Module):
      def __init__(self, in_channel, out_channel, d=[1,3,5,7]):
        super(skip_basic, self).__init__()  
        self.dilated1 = dilated_conv(in_channel, out_channel//4,dilated=d[0])
        self.dilated2 = dilated_conv(in_channel, out_channel//4,dilated=d[1])
        self.dilated3 = dilated_conv(in_channel, out_channel//4,dilated=d[2])
        self.dilated4 = dilated_conv(in_channel, out_channel//4,dilated=d[3])
        self.skip = nn.Sequential(_layer.conv1x1(in_channel, out_channel),
                                  nn.BatchNorm2d(out_channel),)
        self.relu = nn.ReLU(inplace=True)
      def forward(self,x):
            o1 = self.dilated1(x)
            o2 = self.dilated2(x)
            o3 = self.dilated3(x)
            o4 = self.dilated4(x)
            idx = self.skip(x)
            out = torch.cat((o1,o2,o3,o4),1)
            out = out + idx
            out = self.relu(out)
            return out
      
class left_to_right(nn.Module):
      def __init__(self, in_c, out_c, d):
        super(left_to_right, self).__init__()  
        self.conv0 = skip_basic(in_c[0], out_c[0],d[0])
        self.conv1 = skip_basic(in_c[1], out_c[1],d[1])
        self.conv2 = skip_basic(in_c[2], out_c[2],d[2])
        self.conv3 = skip_basic(in_c[3], out_c[3],d[3])
        self.conv4 = skip_basic(in_c[4], out_c[4],d[4])
      def forward(self,r_level0, r_level1, r_level2, r_level3, r_level4):
            skip1_0 = self.conv0(r_level0)
            skip1_1 = self.conv1(r_level1)
            skip1_2 = self.conv2(r_level2)
            skip1_3 = self.conv3(r_level3)
            skip1_4 = self.conv4(r_level4)
            return skip1_0, skip1_1,skip1_2,skip1_3,skip1_4
class skip1(nn.Module):
    def __init__(self, in_channels, out_channels, d):
        super(skip1, self).__init__()  
        self.r_lr = left_to_right(in_channels, out_channels,d)
        self.d_lr = left_to_right(in_channels, out_channels,d)
    def forward(self,r_level0, r_level1, r_level2, r_level3, r_level4,
                      d_level0, d_level1, d_level2, d_level3, d_level4):
#          print(level0.shape, level1.shape, level2.shape, level3.shape, level4.shape)
          skip1_0,skip1_1,skip1_2,skip1_3,skip1_4 = self.r_lr(r_level0, r_level1, r_level2, r_level3, r_level4)
          skip2_0,skip2_1,skip2_2,skip2_3,skip2_4 = self.d_lr(d_level0, d_level1, d_level2, d_level3, d_level4)
          return skip1_0,skip1_1,skip1_2,skip1_3,skip1_4, \
                skip2_0,skip2_1,skip2_2,skip2_3,skip2_4
                
###############################################################################
###############################################################################
def get_skip(in_channels, out_channels, skip_type=0):
     if  skip_type == 1:    
           d=[[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,3,5,7],[1,3,5,7]]
           net = skip1 (in_channels,out_channels,d)
     return net




###############################################################################
###############################################################################
###############################################################################
###############################################################################            
def main():
    x0=torch.rand(2,64,64,64) #随便定义一个输入
    x1=torch.rand(2,64,32,32)
    x2=torch.rand(2,64,16,16)
    x3=torch.rand(2,64,8,8)
    x4=torch.rand(2,64,4,4)
#    net = Encoder (basic_group_layer_bn,basic_encoder_block_cat_attention ,
#                   in_channels, out_channels,layers)
#    net = skip4 (skip_basic, [64,64,64,64,64],[128,128,128,128,128] )
    net = get_skip([64,64,64,64,64,64,64],[64,64,64,64,64,64,64] ,18)
    out = net.forward(x0,x1,x2,x3,x4,x0,x1,x2,x3,x4)
    print(out[0].shape)
#    writer = SummaryWriter(log_dir='./log')
#    
#    writer.add_graph(net, (x0,x1,x2,x3,x4,x0,x1,x2,x3,x4))
#    
#    writer.close()  
    print('done')   
if __name__ == '__main__':
      main()
            