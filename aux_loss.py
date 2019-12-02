# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 10:46:45 2018

@author: Dell
"""
from torch import nn
from tensorboardX import SummaryWriter

###############################################################################
###############################################################################      
class basic_aux_loss(nn.Module):
      def __init__(self, in_channel = 64, out_channel = 1):
            super(basic_aux_loss, self).__init__() 
            self.Aux =nn.Sequential(
                    nn.Conv2d(in_channel, out_channel,kernel_size=3, stride = 1,
                              padding = 1, bias=False),
                    nn.Sigmoid())
      def forward(self,x ):
            out =  self.Aux(x)
            return out  

class basic_aux_loss1(nn.Module):
      def __init__(self, in_channel = 64, out_channel = 1):
            super(basic_aux_loss1, self).__init__() 
            self.Aux =nn.Sequential(
                    nn.Conv2d(in_channel, out_channel,kernel_size=3, stride = 1,
                              padding = 1, bias=False),
                    nn.Sigmoid())
      def forward(self,x ):
            out =  self.Aux(x)
            return out  



class aux_loss8(nn.Module):
      def __init__(self, in_channel, out_channel = 1, loss = basic_aux_loss):
            super(aux_loss8, self).__init__() 
            self.aux0 = loss(in_channel[0], out_channel)
            self.aux1 = loss(in_channel[1], out_channel)
            self.aux2 = loss(in_channel[2], out_channel)     
            self.aux3 = loss(in_channel[3], out_channel)
            self.aux4 = loss(in_channel[4], out_channel)     
            self.aux5 = loss(in_channel[5], out_channel)
            self.aux6 = loss(in_channel[6], out_channel)
            self.aux7 = loss(in_channel[7], out_channel)
            self.aux8 = loss(in_channel[8], out_channel)     
            self.aux9 = loss(in_channel[9], out_channel)
            self.aux10 = loss(in_channel[10], out_channel)     
            self.aux11 = loss(in_channel[11], out_channel)
      def forward(self, o1_1, o1_2, o1_3, o2_1, o2_2, o2_3, o3_1, o3_2, o3_3, o4_1, o4_2, o4_3):
#            print(ax0.shape, ax1.shape, ax2.shape, ax1.shape, ax2.shape)
            au0 = self.aux0(o1_1)
            au1 = self.aux1(o1_2)
            au2 = self.aux2(o1_3)
            au3 = self.aux3(o2_1)
            au4 = self.aux4(o2_2)
            au5 = self.aux5(o2_3)
            au6 = self.aux6(o3_1)
            au7 = self.aux7(o3_2)
            au8 = self.aux8(o3_3)
            au9 = self.aux9(o4_1)
            au10 = self.aux10(o4_2)
            au11 = self.aux11(o4_3)

            return au0, au1, au2, au3, au4, au5,au6, au7, au8, au9, au10, au11

###############################################################################
###############################################################################          
def get_aux_loss(in_channel, out_channel=1, aux_type=0):
     if aux_type == 1:
           net = aux_loss8 (in_channel, out_channel,basic_aux_loss1)

     return net           
###############################################################################
###############################################################################
###############################################################################            
def main():
    x3=torch.rand(2,128,16,16) #随便定义一个输入  
    x2=torch.rand(2,128,32,32) #随便定义一个输入
    x1=torch.rand(2,128,64,64) #随便定义一个输入
    x0=torch.rand(2,128,128,128) #随便定义一个输入
    

#    net = Encoder (basic_group_layer_bn,basic_encoder_block_cat_attention ,
#                   in_channels, out_channels,layers)
    net =aux_loss2(128, 6)
    net.forward(x0,x1)
    writer = SummaryWriter(log_dir='./log')
    
    writer.add_graph(net, (x0,x1))
    
    writer.close()  
    print('done')   
if __name__ == '__main__':
#      main()   
      print(type((1,2,3)))