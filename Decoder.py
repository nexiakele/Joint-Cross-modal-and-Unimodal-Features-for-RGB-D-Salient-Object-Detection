# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:18:07 2018

@author: Dell
"""
import torch
from torch import nn
from tensorboardX import SummaryWriter
import layers as _layers
import torch.nn.functional as F
################################################################################################
################################################################################################
################################################################################################
class right_to_left_head_a(nn.Module):
      def __init__(self, args, in_channel, out_c, stride = 1):
            super(right_to_left_head_a, self).__init__()
            self.r_d = _layers.BasicBlock(in_channel, out_c[0])
            self.r = _layers.BasicBlock(in_channel, out_c[1])
            self.d =  _layers.BasicBlock(in_channel, out_c[1])
      def forward(self,rgb_small, depth_small):
            r_d = rgb_small + depth_small
            rd  = self.r_d(r_d)
            rr  = self.r(rgb_small)
            dd  = self.d(depth_small)
            return rd, rr, dd
class right_to_left_body_a(nn.Module):
      def __init__(self, args,inc1, inc2 ,out_c):
            super(right_to_left_body_a, self).__init__() 
            up_sample = args['up_sample']
            in_c1,in_c2,in_c3, in_c4 = inc1[0], inc1[1], inc1[2], inc2
            self.up_conv1 = up_sample(in_c1, in_c1)
            self.up_conv2 = up_sample(in_c2, in_c2)
            self.up_conv3 = up_sample(in_c3, in_c3)
            in_channel = in_c1+in_c2+in_c4
            self.l_r_d = _layers.BasicBlock(in_channel, out_c[0])
            in_channel=in_c2+in_c4
            self.r = _layers.BasicBlock(in_channel, out_c[1])
            in_channel=in_c3+in_c4
            self.d = _layers.BasicBlock(in_channel, out_c[2])
      def forward(self,lrd, lrr, ldd, nr, nd):
            lrd = self.up_conv1(lrd)
            lrr = self.up_conv2(lrr)
            ldd = self.up_conv3(ldd)
            
            nrd = nr + nd
            lrr_dd  = lrr + ldd
            frd = torch.cat((lrd, lrr_dd, nrd),1)
            frd = self.l_r_d(frd)
            
            rr = torch.cat((lrr, nr),1)
            rr  = self.r(rr)
            dd = torch.cat((ldd, nd), 1)
            dd = self.d(dd)
            return frd, rr, dd
class Aggregation_right_to_left_4(nn.Module):
      def __init__(self, args, in_cs,  out_cs):
            super(Aggregation_right_to_left_4, self).__init__() 
            self.body4 = right_to_left_head_a(args,in_cs[4],out_cs[4])
            self.body3 = right_to_left_body_a(args,out_cs[4],in_cs[3], out_cs[3])
            self.body2 = right_to_left_body_a(args,out_cs[3],in_cs[2], out_cs[2])
            self.body1 = right_to_left_body_a(args,out_cs[2],in_cs[1], out_cs[1])
            self.body0 = right_to_left_body_a(args,out_cs[1],in_cs[0], out_cs[0])
      def forward(self,rgb_level0, rgb_level1, rgb_level2, rgb_level3, rgb_level4,
                        depth_level0, depth_level1, depth_level2, depth_level3, depth_level4):
            rd4, rr4, dd4 = self.body4(rgb_level4, depth_level4)
            rd3, rr3, dd3 = self.body3(rd4, rr4, dd4, rgb_level3, depth_level3)
            rd2, rr2, dd2 = self.body2(rd3, rr3, dd3, rgb_level2, depth_level2)
            rd1, rr1, dd1 = self.body1(rd2, rr2, dd2, rgb_level1, depth_level1)
            rd0, rr0, dd0 = self.body0(rd1, rr1, dd1, rgb_level0, depth_level0)
            return rd0, rr0, dd0, rd1, rr1, dd1, rd2, rr2, dd2, rd3, rr3, dd3, rd4, rr4, dd4

def get_decoder(in_channels, out_channels, decoder_type):
##########################使用第一层的特征#######################################
      if  decoder_type ==1:
            args={'up_sample': _layers.basic_interpolate_up}
            net = Aggregation_right_to_left_4 (args , in_channels, out_channels)
   
      return net
      

def main():
##############################################################################
#    x4=torch.rand(2,128,8,8) #随便定义一个输入  
#    x3=torch.rand(2,128,16,16) #随便定义一个输入  
#    x2=torch.rand(2,128,32,32) #随便定义一个输入
#    x1=torch.rand(2,128,64,64) #随便定义一个输入
#    x0=torch.rand(2,128,128,128) #随便定义一个输入
#    net = get_decoder_aux(128, 128, 11)
##############################################################################
    x4=torch.rand(2,64,8,8) #随便定义一个输入  
    x3=torch.rand(2,64,16,16) #随便定义一个输入  
    
    
    x2=torch.rand(2,64,32,32) #随便定义一个输入
    maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
    xx2 = maxpool(x2)
    print(x2.shape)
    print(xx2.shape)
    
    x1=torch.rand(2,64,64,64) #随便定义一个输入
    x0=torch.rand(2,64,128,128) #随便定义一个输入
    
    d4=torch.rand(2,64,8,8) #随便定义一个输入  
    d3=torch.rand(2,64,16,16) #随便定义一个输入  
    d2=torch.rand(2,64,32,32) #随便定义一个输入
    d1=torch.rand(2,64,64,64) #随便定义一个输入
    d0=torch.rand(2,64,128,128) #随便定义一个输入
    in_channels = [64,64,64,64,64,64 ]
    out_channels = [64,64,64,64,64,64]
#    #####################dense###############################################
#    net = get_decoder(in_channels, out_channels,18)
#    #####################no_dense##############################################
##    net = get_decoder_aux([128,128, 128,128],[128,128, 128,128],  decoder_type)
##    net.forward(x0,x1,x2,x3, x4)
#    out = net.forward(x0,x1,x2,x3, x4,d0,d1,d2,d3, d4)
#    print(len(out))
#    writer = SummaryWriter(log_dir='./log')
##    writer.add_graph(net, (x0,x1,x2,x3, x4))
#    writer.add_graph(net, (x0,x1,x2,x3, x4,x0,x1,x2,x3, x4))
#    writer.close()  
#    print('done')   
if __name__ == '__main__':
      main()      