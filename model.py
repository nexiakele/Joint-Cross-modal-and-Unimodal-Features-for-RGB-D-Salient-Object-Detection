# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:13:27 2018

@author: Dell
"""

import torch
from torch import nn
from tensorboardX import SummaryWriter
import numpy as np
import Encoder 
import skip  
import Decoder
import aux_loss
import predict
#from torchsummary import summary

###############################################################################
###########################model###############################################
class model8_3(nn.Module):
    def __init__(self,channels,encoder_type=1,skip_type=1,
                 decoder_type=2,aux_type=0,predict_type=0, init_weights=True):
        super(model8_3, self).__init__() 
#        #######################################################################
        in_channels = channels['skip_in_channels']
        out_channels = channels['skip_out_channels']
        self.skip     = skip.get_skip(in_channels, out_channels, skip_type)
#        #######################################################################
        in_channels = channels['decoder_in_channels']
        out_channels = channels['decoder_out_channels']
        self.Decoder  = Decoder.get_decoder(in_channels,out_channels, decoder_type)
#        #######################################################################
        in_channels = channels['aux_in_channels']
        out_channels = channels['aux_out_channels']
        self.aux      = aux_loss.get_aux_loss(in_channels,out_channels,aux_type)
        in_channels = channels['predict_in_channels']
        self.predict  = predict.get_predict(in_channels, predict_type)
        #######################################################################
        if init_weights:
            print('参数初始化')
            self._initialize_weights()
        #######################################################################
        in_channels = channels['encoder_in_channels']
        out_channels = channels['encoder_out_channels']
        layers  = channels['encoder_layes']
        print('载入VGG')
        self.Encoder  = Encoder.get_encoder(in_channels, out_channels, layers, encoder_type)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self,x,y): 
        feature1_0, feature1_1, feature1_2, feature1_3, feature1_4,   \
        feature2_0, feature2_1, feature2_2, feature2_3, feature2_4= self.Encoder(x,y)
#        print(feature0.shape, feature1.shape, feature2.shape, feature3.shape, feature4.shape)
        skip1_0,skip1_1,skip1_2,skip1_3,skip1_4, \
        skip2_0,skip2_1,skip2_2,skip2_3,skip2_4 = self.skip(feature1_0, feature1_1, feature1_2, feature1_3, feature1_4,
                                                           feature2_0, feature2_1, feature2_2, feature2_3, feature2_4)
#        print(skip1_0.shape,skip1_1.shape,skip1_2.shape,skip1_3.shape,
#                  skip1_4.shape,skip2_0.shape,skip2_1.shape,skip2_2.shape,skip2_3.shape,skip2_4.shape)
        rd0, rr0, dd0, rd1, rr1, dd1, rd2, rr2, dd2, rd3, rr3, dd3, rd4, rr4, dd4  = self.Decoder(skip1_0,skip1_1,skip1_2,skip1_3,skip1_4,
                                                                                                     skip2_0,skip2_1,skip2_2,skip2_3,skip2_4)
        
        au1_1, au1_2, au1_3,au2_1, au2_2, au2_3,au3_1, au3_2, au3_3, au4_1, au4_2, au4_3=self.aux(rd1, rr1, dd1, rd2, rr2, dd2, rd3, rr3, dd3, rd4, rr4, dd4)
        out,au0_1, au0_2, au0_3 = self.predict(rd0, rr0, dd0)
        return (out,au0_1, au0_2, au0_3),(au1_1,au1_2,au1_3),(au2_1,au2_2,au2_3),(au3_1,au3_2,au3_3),(au4_1,au4_2,au4_3)

###############################################################################
###############################################################################
###############################################################################
def get_model(model_tpye = 0):
      channels = {
                  ##########Encoder
                  'encoder_in_channels' :  [1,  64,  128,  256,  512],
                  'encoder_out_channels' : [64, 128, 256,  512,  512],
                  'encoder_layes' : [1,  2,  2,  2,  2],
                  ##########skip
                  'skip_in_channels' : [64, 128, 256,  512,  512],
                  'skip_out_channels' : [48, 128, 256, 512, 512],
                  ##########decoder
                  'decoder_in_channels' :  [48, 128, 256, 512, 512],
                  'decoder_out_channels' : [[96,96,96], [128,128,128],[256,256,256],[384,384,384],[512,512,512]],
                  ##########aux_loss
                  'aux_in_channels' :  [128,128,128,256,256,256,384,384,384,512,512,512],
                  'aux_out_channels' :  1,
                  ##########predict
                  'predict_in_channels' : [96,96,96]
                  }
      if  model_tpye == 1:
           net = model8_3(channels,1, 1, 1, 1, 1)
      return net     

###############################################################################
###############################################################################
###############################################################################      
import Loss
def main():

    x=torch.rand(2,3,64,64) #随便定义一个输入
    y=torch.rand(2,1,64,64) #随便定义一个输入
    
    gt=torch.rand(2,1,32,32) #随便定义一个输入
    
    net =  get_model(1)

    out = net.forward(x,y)
    print('---------------')
#    print(final_out.shape,out.shape, out1.shape, out2.shape, out3.shape, out4.shape)
#    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape,out[5].shape, out[6].shape)
#    print(out[7].shape, out[8].shape, out[9].shape, out[10].shape, out[11].shape,out[12].shape, out[13].shape, out[14].shape)
    out0 = out[0]
    for o in out:
        if isinstance(o, tuple):
            for i in o:
                print(i.shape)
        else:
            print(o.shape)        
#    writer = SummaryWriter(log_dir='./log')
#    
#    writer.add_graph(net, (x,y))
#    
#    writer.close()  
    print('done')

if __name__ == '__main__':
    #test()
    main()