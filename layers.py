# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 10:28:14 2018

@author: Dell
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter

###############################################################################
###############################################################################
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv1x1xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
def conv1x1xbn(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                   padding=0, bias=False),
                         nn.BatchNorm2d(out_planes))
###############################################################################
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv3x3xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
def conv3x3xbn(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_planes))
          
def convkxk(in_planes, out_planes,k_size=5, stride=1):
    """kxk convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=k_size, stride=stride,
                     padding=k_size//2, bias=False)
def convkxkxbnxrelu(in_planes, out_planes,k_size=5, stride=1):
    """kxk convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=k_size, stride=stride,
                                   padding=k_size//2, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())
def convkxkxbn(in_planes, out_planes,k_size=5, stride=1):
    """kxk convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=k_size, stride=stride,
                                   padding=k_size//2, bias=False),
                         nn.BatchNorm2d(out_planes))
def sep_convkxkxbnxrelu(in_planes,k_size=5, stride=1):
    """kxk convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, in_planes, kernel_size=k_size, stride=stride,
                                   padding=k_size//2,groups=in_planes, bias=False),
                         nn.BatchNorm2d(in_planes),
                         nn.ReLU())
class split_tensor(nn.Module):
      def __init__(self, blocks):
        super(split_tensor, self).__init__()  
        self.blocks = blocks
      def forward(self,x):
             layers = []
             for i in range(self.blocks):
                   mask = x[:, i:i+1, :, :]
                   layers.append(mask)
             return layers
###############################################################################
###############################################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,group_number=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes,
                                   kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(planes))
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu2(out)
        return out
  
class baseblock(nn.Module):
    def __init__(self, in_channels, out_channel, is_d = False):
        super(baseblock, self).__init__()
        d2, d3 = 1, 1
        if is_d:
              d2, d3 = 2, 3
        self.conv1 = nn.Sequential(
                        conv3x3(in_channels, out_channel//4),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channel//4, out_channel//2,kernel_size=3, stride = 1,
                                  padding = d2,dilation=d2, bias=False),
                        nn.BatchNorm2d(out_channel//2),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channel//2, out_channel//4,kernel_size=3, stride = 1,
                                  padding = d3,dilation=d3, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channel,
                                   kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())
    def forward(self, x):
        identity = self.downsample(x)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        out = torch.cat((o1,o2,o3), 1)
        out = out + identity
        return out



class baseblock1(nn.Module):
    def __init__(self, in_channels, out_channel, d = [1, 1, 1 ]):
        super(baseblock1, self).__init__()
        d1, d2, d3 = d[0], d[1], d[2]
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=3, stride = 1,
                                  padding = d1,dilation=d1, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channel//4, out_channel//2,kernel_size=3, stride = 1,
                                  padding = d2,dilation=d2, bias=False),
                        nn.BatchNorm2d(out_channel//2),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channel//2, out_channel//4,kernel_size=3, stride = 1,
                                  padding = d3,dilation=d3, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channel,
                                   kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channel),
                                   nn.ReLU())
    def forward(self, x):
        identity = self.downsample(x)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        out = torch.cat((o1,o2,o3), 1)
        out = out + identity
        return out


class baseblock2(nn.Module):
    def __init__(self, in_channels, out_channel, d = [1, 1, 1 ]):
        super(baseblock2, self).__init__()
        d1, d2, d3 = d[0], d[1], d[2]
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=3, stride = 1,
                                  padding = d1,dilation=d1, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channel//4, out_channel//2,kernel_size=3, stride = 1,
                                  padding = d2,dilation=d2, bias=False),
                        nn.BatchNorm2d(out_channel//2),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channel//2, out_channel//4,kernel_size=3, stride = 1,
                                  padding = d3,dilation=d3, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.downsample = None
        if in_channels != out_channel:          
            self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channel,
                                       kernel_size=1, stride=1, bias=False),
                                       nn.BatchNorm2d(out_channel),
                                       nn.ReLU())
    def forward(self, x):
        identity = x
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        out = torch.cat((o1,o2,o3), 1)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out


class inception(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(inception, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=1, stride = 1,
                                  padding = 0,bias=False),
                        nn.BatchNorm2d(out_channel//4),)
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=3, stride = 1,
                                  padding = 1, bias=False),
                        nn.BatchNorm2d(out_channel//4),)
        self.conv5 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=5, stride = 1,
                                  padding = 2, bias=False),
                        nn.BatchNorm2d(out_channel//4),)
        self.conv7 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=7, stride = 1,
                                  padding = 3, bias=False),
                        nn.BatchNorm2d(out_channel//4),)
        self.downsample = nn.Sequential(nn.Conv2d(in_channels, out_channel,
                                   kernel_size=1, stride=1, bias=False),
                                   nn.BatchNorm2d(out_channel),)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = self.downsample(x)
        o1 = self.conv1(x)
        o2 = self.conv3(x)
        o3 = self.conv5(x)
        o4 = self.conv7(x)
        out = torch.cat((o1,o2,o3, o4), 1)
        out = out + identity
        out = self.relu(out)
        return out
  
class inception_v2(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(inception_v2, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=1, stride = 1,
                                  padding = 0,bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=3, stride = 1,
                                  padding = 1, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.conv5 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=5, stride = 1,
                                  padding = 2, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())
        self.conv7 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channel//4,kernel_size=7, stride = 1,
                                  padding = 3, bias=False),
                        nn.BatchNorm2d(out_channel//4),
                        nn.ReLU())

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv3(x)
        o3 = self.conv5(x)
        o4 = self.conv7(x)
        out = torch.cat((o1,o2,o3, o4), 1)
        return out
###############################################################################
###############################################################################
class basic_subpixel_up(nn.Module):
    def __init__(self, in_channel, out_channel, scale = 2):
        super(basic_subpixel_up, self).__init__() 
        self.deconv = nn.Sequential(
                        nn.Conv2d(in_channel, out_channel  * scale * scale ,
                                           kernel_size=1, stride = 1,padding = 0, bias=False),
                        nn.PixelShuffle(scale),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU())
    def forward(self,x):
        out = self.deconv(x)
        return out
###############################################################################
###############################################################################
class basic_deconv_up(nn.Module):
    def __init__(self, in_channel, out_channel=None, scale = 2):
        super(basic_deconv_up, self).__init__() 
        if out_channel is None:
              out_channel = in_channel
        self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(in_channel, out_channel,kernel_size=3, stride = 2,
                                           padding = 1,output_padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU())
    def forward(self,x):
        out = self.deconv(x)
        return out
class basic_deconv_up2(nn.Module):
    def __init__(self, in_channel, out_channel=None, scale = 2):
        super(basic_deconv_up2, self).__init__() 
        if out_channel is None:
              out_channel = in_channel
        self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(in_channel, out_channel,kernel_size=2, stride = 2,
                                           padding = 0, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU())
    def forward(self,x):
        out = self.deconv(x)
        return out
class basic_interpolate_up(nn.Module):
    def __init__(self, in_channel, out_channel=None, scale = 2):
        super(basic_interpolate_up, self).__init__() 
        self.scale = scale
    def forward(self,x):
        out = F.interpolate(x, 
                            scale_factor=self.scale, 
                            mode='bilinear', 
                            align_corners=False)
        return out
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)




class ContextBlock2d(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out


class ContextBlock2d2(nn.Module):

    def __init__(self, inplanes, planes, pool, fusions):
        super(ContextBlock2d2, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.tanh(self.channel_mul_conv(context))
            out = x * channel_mul_term + x
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out



class selection(nn.Module):

    def __init__(self, inplanes, planes):
        super(selection, self).__init__()

        self.inplanes = inplanes
        self.planes = planes

        self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_mul_conv = nn.Sequential(
                            nn.Conv2d(self.inplanes*2, self.planes, kernel_size=1),
                            nn.LayerNorm([self.planes, 1, 1]),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(self.planes, self.inplanes, kernel_size=1))
        
        self.adapool = nn.AdaptiveAvgPool2d(1)
    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        # [N, C, 1, 1] spatial context
        context1 = self.spatial_pool(x)
        # [N, C, 1, 1] gobal context
        context2 = self.adapool(x)
        #[2N, C, 1, 1]
        context = torch.cat((context1, context2),1)
        channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
        out = x * channel_mul_term
        return out
def main():
      
      
    x=torch.rand(2,128,128,128) #随便定义一个输入

#    net = basic_group_layer_bn(256, 128)
#    net = basic_subpixel_interpolate_multi_maxpool2(64, 64)
    net = BasicBlock2(128,128)
    
    writer = SummaryWriter(log_dir='./log')
    writer.add_graph(net, (x))
    writer.close()  
    print('done')   
if __name__ == '__main__':
      main()