# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:14:35 2018

@author: Dell
"""
import torch
from torch import nn
from tensorboardX import SummaryWriter
import layers as _layers
from torchvision.models import vgg16_bn, resnet18, resnet34, resnet152
###############################################################################
############################Encoder############################################
###############################################################################
'''
###############################################################################
input:
      #########################################################################
      encoder_head: block0的结构
               Encoder_head_base: 对于block0的信息不做处理
               Encoder_head_add : block0的特征进行相加
               Encoder_head_cat ：block0的特征进行级联
      #########################################################################
      block：除block0以外的block的结构
              basic_encoder_block      ： 对rgb, depth的分支进行单独处理没有交互
              basic_encoder_block_add  ： 对rgb,depth的特征进行相加处理
              Encoder_head_cat         ： 对rgb,depth的特征进行级联处理
              basic_encoder_block_add_attention ： 对rgb,depth的特征进行attenion并相加处理
              basic_encoder_block_cat_attention ： 对rgb,depth的特征进行attenion并级联处理
      #########################################################################
      encoder_basic_block：整个encoder中使用的基本结构
              basic_group_layer_bn ： ResneXt中的基本的残差块
      #########################################################################
      in_channels ： 数组结构为[block0_in_channels, block1_in_channels...]
      out_channels:  数组结构为[block0_out_channels, block1_out_channels...]
      layers      :  每个block的层数[block0_layers, block1_layers]
###############################################################################
'''
class Vgg16(torch.nn.Module):
    def __init__(self,is_depth = False, pretrained = True):
        super(Vgg16, self).__init__()
        features = list(vgg16_bn(pretrained=pretrained).features)[:43]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        if is_depth:
              features[0] = nn.Conv2d(1, 64,kernel_size=1, stride = 1,padding = 0, bias=False)
        self.features = nn.ModuleList(features).eval() 
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {5,12,22,32}:
                results.append(x)
        return results[0], results[1],results[2],results[3], x

class Vgg16_2(torch.nn.Module):
    def __init__(self,is_depth = False, pretrained = True):
        super(Vgg16_2, self).__init__()
        features = list(vgg16_bn(pretrained=pretrained).features)[:43]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        if is_depth:
              features[0] = nn.Conv2d(1, 64,kernel_size=3, stride = 1,padding = 1, bias=False)
        self.features = nn.ModuleList(features).eval() 
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {5,12,22,32}:
                results.append(x)
        return results[0], results[1],results[2],results[3], x

class resnet34_(torch.nn.Module):
    def __init__(self,is_depth = False, pretrained = True):
        super(resnet34_, self).__init__()
        features = list(resnet34(pretrained=pretrained).children())[0:-2]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        features[0].stride = 1
        if is_depth:
              features[0] = nn.Conv2d(1, 64,kernel_size=7, stride = 1,padding = 3, bias=False)
        self.features = nn.ModuleList(features).eval() 
        print(features[2])
    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {2,4,5,6}:
                results.append(x)
        return results[0], results[1],results[2],results[3], x

class resnet152_(torch.nn.Module):
    def __init__(self,is_depth = False, pretrained = True):
        super(resnet152_, self).__init__()
        features = list(resnet152(pretrained=pretrained).children())[0:-2]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        features[0].stride = 1
        if is_depth:
              features[0] = nn.Conv2d(1, 64,kernel_size=7, stride = 1,padding = 3, bias=False)
        self.features = nn.ModuleList(features).eval() 

    def forward(self, x):
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {2,4,5,6}:
                results.append(x)
        return results[0], results[1],results[2],results[3], x		
		
class Encoder_vgg(torch.nn.Module):
    def __init__(self,pretrained = False):
          super(Encoder_vgg, self).__init__()
          self.rgb_vgg = Vgg16(pretrained=pretrained)
          self.depth_vgg = Vgg16(is_depth=True,pretrained=pretrained)
    def forward(self, rgb, depth):
          rgb_x0 , rgb_x1, rgb_x2, rgb_x3, rgb_x4 = self.rgb_vgg(rgb)
          depth_x0, depth_x1, depth_x2, depth_x3, depth_x4 = self.depth_vgg(depth)
          return rgb_x0 , rgb_x1, rgb_x2, rgb_x3, rgb_x4, \
                depth_x0, depth_x1, depth_x2, depth_x3, depth_x4
                
class Encoder_resnet34(torch.nn.Module):
    def __init__(self,pretrained = True):
          super(Encoder_resnet34, self).__init__()
          self.rgb_vgg = resnet34_(pretrained=True)
          self.depth_vgg = resnet34_(is_depth=True,pretrained=True)
    def forward(self, rgb, depth):
          rgb_x0 , rgb_x1, rgb_x2, rgb_x3, rgb_x4 = self.rgb_vgg(rgb)
          depth_x0, depth_x1, depth_x2, depth_x3, depth_x4 = self.depth_vgg(depth)
          return rgb_x0 , rgb_x1, rgb_x2, rgb_x3, rgb_x4, \
                depth_x0, depth_x1, depth_x2, depth_x3, depth_x4
###############################################################################
###############################################################################        
'''
###############################################################################
############################函数名##############################################
get_encoder ： 取得对应于encoder_type的类型
###############################################################################
inputs:
      #########################################################################
      in_channels: 输入的数据
      out_channels:输出
      layer:每个block的层数
      encoder_type: encoder的类型：
            ###################################################################
            0 :  特征相加没有attention
            1 :  特征级联没有attention
            2 :  特征相加有 attention
            3:   特征级联有 attention
###############################################################################
###############################################################################
'''
def get_encoder(in_channels,out_channels,layers,encoder_type  = 0):
      flag = False
      if encoder_type ==1:
            net = Encoder_vgg(flag)
      elif encoder_type ==2:
            net = Encoder_resnet34(flag)
      elif encoder_type ==3:
            net = Vgg16(is_depth=False, pretrained=flag)
      elif encoder_type ==3:
            net = Vgg16(is_depth=False, pretrained=flag)
      elif encoder_type ==4:
            net = Vgg16(is_depth=True, pretrained=flag)      
      return net
  
def main():
    x=torch.rand(2,96,64,64) #随便定义一个输入
    y=torch.rand(2,64,64,64) #随便定义一个输入
    mix=torch.rand(2,64,64,64) #随便定义一个输入
    rgb=torch.rand(2,3,64,64) #随便定义一个输入
    depth=torch.rand(2,1,64,64) #随便定义一个输入
    in_channels =  [1,   64, 128, 256, 512]
    out_channels = [64, 128, 256, 512, 512]
    layers = [2,2,3,3,3]
    
#    net = get_encoder(in_channels,out_channels,layers, 1 )
    net= resnet34_()
#    net.forward(rgb)
    
#    args = {'head':Encoder_head,
#            'base_encoder_block':basic_encoder_block,
#            'base_block' : basic_residual_block,}
#    net = get_encoder(in_channels,out_channels,layers,encoder_type = 3)
#    net.forward(x,y)
#    net = base_block2(None,128,128)#basic_encoder_block(base_block2,128,128,3)
#    net = Encoder_head(None,1, 64)
#    out = net.forward(x,y,mix)
#    out = net.forward(rgb, depth)
#    net = basic_block2(None,128,256, 2)
#    net = basic_encoder_block(None,basic_residual_block, 64, 128, 3)
#    net = Encoder(args, in_channels, out_channels, layers)
#    out = net.forward(rgb,depth)
#    writer = SummaryWriter(log_dir='./log')
#    
#    writer.add_graph(net, (rgb,depth))
#    
#    writer.close()  
#    print('done')   
if __name__ == '__main__':
      main()