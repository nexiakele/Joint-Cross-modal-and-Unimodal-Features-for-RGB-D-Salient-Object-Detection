# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:13:27 2018

@author: Dell
"""
import tools.rgb_loader as nda
from tools.Tools import load_ckpt
import torch
from torch import nn
from tensorboardX import SummaryWriter
import numpy as np
import Encoder 
import skip  
import Decoder
import aux_loss
import predict
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage import io
def save_tensor(im_tensor, path, name=None, mode='one'):
    out = im_tensor.cpu().clone()
    out = out.detach().numpy()
    print(out.shape)
#    print(out.shape)
    if mode == 'one':
        io.imsave(path+'/'+name+'.png',out[0][0])
    elif mode == 'many':
        count = 0
        for i in out[0]:
            print(count, i.shape)
            count = count+1
            io.imsave(path+'/'+str(count)+'.png',i)
def make_file(name):
    path = 'visual'
    if not os.path.exists(path):
        os.mkdir(path)
    path = 'visual/' + name
    if not os.path.exists(path):
        os.mkdir(path)
    rp = path + '/r'
    dp = path + '/d'
    rdp = path + '/rd'
    if not os.path.exists(rp):
        os.mkdir(rp)
    if not os.path.exists(dp):
        os.mkdir(dp)
    if not os.path.exists(rdp):
        os.mkdir(rdp)
    op = path + '/out'
    if not os.path.exists(op):
        os.mkdir(op)
    return rp, dp, rdp, op  

def make_file2(name):
    path = 'visual'
    if not os.path.exists(path):
        os.mkdir(path)
    path = 'visual/' + name
    if not os.path.exists(path):
        os.mkdir(path)
    rp = path + '/out'
    if not os.path.exists(rp):
        os.mkdir(rp)
    return rp
def save_feature_map(out1, out2,  r_b, d_b,rd_b, name):
    rp, dp, rdp, op  = make_file(name)
    save_tensor(out1, op, 'p', 'one')
    save_tensor(out2, op, 'rd', 'one')
    save_tensor(r_b, rp, mode='many')
    save_tensor(d_b, dp, mode='many')
    save_tensor(rd_b, rdp, mode='many')
def save_tensor2(im_tensor, path, name, mode='one'):
    out = im_tensor.cpu().clone()
    out = out.detach().numpy()
    print(out.shape)
#    print(out.shape)
    if mode == 'one':
        io.imsave(path+'/'+name+'.png',out[0][0])
    elif mode == 'many':
        count = 0
        for i in out[0]:
            count = count+1
            io.imsave(path+'/'+str(count)+'_'+name+'.png',i)    
def save_feature_map2(out1, out2,  r_b, d_b,rd_b, name, weight=None):
    rp, dp, rdp, op  = make_file(name)
    save_tensor2(out1, op, 'p', 'one')
    save_tensor2(out2, op, 'rd', 'one')
    save_tensor2(r_b, rp,'r', mode='many')
    save_tensor2(d_b, rp, 'd',  mode='many')
    save_tensor2(rd_b, rp,'rd', mode='many')  
    if weight is not None:
       weight = torch.squeeze(weight)
       weight = weight.cpu().clone() 
       weight = weight.detach().numpy()
       np.savetxt(op+'weight.txt', weight )
       
def save_tensor3(im_tensor, path, name, mode='one', weight=None):
    out = im_tensor.cpu().clone()
    out = out.detach().numpy()
#    print(out.shape)
    if mode == 'one':
        io.imsave(path+'/'+name+'.png',out[0][0])
    elif mode == 'many':
        count = 0
        for i, w in zip(out, weight):
#            print(count, i.shape)
            count = count+1
            io.imsave(path+'/'+str(count)+'_'+name+ str(w)+'.png',i)  
def save_feature_map3(out1, out2,  r_b, d_b, rd_b, name, weight=None):
    rp, dp, rdp, op  = make_file(name)
    if weight is not None:
       weight = torch.squeeze(weight)
       weight = weight.cpu().clone() 
       weight = weight.detach().numpy()
       print(weight.max(), weight.mean())
       (wrd, wr, wd) = np.split(weight, 3)
    save_tensor3(out1, op, 'p', 'one')
    save_tensor3(out2, op, 'rd', 'one')
    save_tensor3(r_b, rp,'r', mode='many',  weight=wr)
    save_tensor3(d_b, rp, 'd',  mode='many',weight=wd)
    save_tensor3(rd_b, rp,'rd', mode='many', weight=wrd)  

def save_tensor10(im_tensor, path, name, mode='one'):
    out = im_tensor.cpu().clone()
    out = out.detach().numpy()
#    print(out.shape)
    if mode == 'one':
        io.imsave(path+'/'+name+'.png',out[0][0])
    elif mode == 'many':
        count = 0
        for i in out:
            print(count, i.shape)
            count = count+1
            io.imsave(path+'/'+str(count)+'_'+name+'.png',i)      
def save_feature_map4(name, fr, fd, frd):
    rp, dp, rdp, op  = make_file(name)
    save_tensor10(fr, rp,'r', mode='many')
    save_tensor10(fd, dp, 'd',  mode='many')
    save_tensor10(frd, rdp,'rd', mode='many')  

def save_tensor4(im_tensor, path, name, mode='one'):
    out = im_tensor.cpu().clone()
    out = out.detach().numpy()
    if mode == 'one':
        io.imsave(path+'/'+name+'.png',out[0][0])
    elif mode == 'many':
        for m, n in zip(out, name):
            io.imsave(path+'/'+n+ '.png',m)  
    
def save_mask(out, out_rd, out_r, out_d, masks, name):
    rp = make_file2(name)
    masks = torch.squeeze(masks)
    save_tensor4(out, rp, 'p')
    save_tensor4(out_rd, rp, 'rd')
    save_tensor4(out_r, rp, 'r')
    save_tensor4(out_d, rp, 'd')
    save_tensor4(masks, rp, ['rd_w', 'r_w', 'd_w'], mode='many')
    
    
def process_weight(weight):
   weight = torch.squeeze(weight)
   weight = weight.cpu().clone() 
   weight = weight.detach().numpy()    
   (wrd, wr, wd) = np.split(weight, 3)
   return wrd, wr, wd
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
        out,au0_1, au0_2, au0_3= self.predict(rd0, rr0, dd0)
        return rd0, rr0, dd0

class compare_model2(nn.Module):
    def __init__(self,channels,encoder_type=1,skip_type=1,
                 decoder_type=2,aux_type=0,predict_type=0, init_weights=True):
        super(compare_model2, self).__init__() 
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
#        self.predict  = predict.get_predict(in_channels, predict_type)
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
        skip1_0,skip1_1,skip1_2,skip1_3,skip1_4 = self.skip(feature1_0, feature1_1, feature1_2, feature1_3, feature1_4,
                                                           feature2_0, feature2_1, feature2_2, feature2_3, feature2_4)
        out0, out1, out2, out3, out4 = self.Decoder(skip1_0,skip1_1,skip1_2,skip1_3,skip1_4)
        
        au0,au1, au2, au3, au4=self.aux(out0, out1, out2, out3, out4)
        b,c,_,_ = skip1_0.shape
        return au0,au1, skip1_4[:,0:c//2,:,:],skip1_4[:,c//2:,:,:],out0
##############################################################################
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
                  'skip_out_channels' : [128,128,128,128,128],
                  ##########decoder
                  'decoder_in_channels' :  [128,128,128,128,128,128,128],
                  'decoder_out_channels' : [128,128,128,128,128,128,128],
                  ##########aux_loss
                  'aux_in_channels' :  [128,128,128,128,128,128,128],
                  'aux_out_channels' :  1,
                  ##########predict
                  'predict_in_channels' : 128
                  }
      if model_tpye == 97:
            channels['skip_out_channels'] =    [48, 128, 256, 384, 384]
            channels['decoder_in_channels'] =  [48, 128, 256, 384, 384]
            channels['decoder_out_channels'] = [[96,96,96], [128,128,128],[192,192,192],[256,256,256],[256,256,256]]
            channels['aux_out_channels'] =  1
            temp = np.array(channels['decoder_out_channels'])
            channels['predict_in_channels'] = [96,96,96, temp.sum()]
            channels['aux_in_channels'] = [128,128,128,192,192,192,256,256,256,256,256,256]
            net = model8_3(channels,1, 6, 23, 4, 63)
      elif model_tpye == 102:
            channels['skip_out_channels'] =    [64, 128, 192, 256, 384]
            channels['decoder_in_channels'] =  [64, 128, 192, 256, 384]
            channels['decoder_out_channels'] = [96, 128, 192, 256, 256]
            channels['aux_out_channels'] =  1
            channels['aux_in_channels'] = [96, 128, 192, 256, 256]
            net = compare_model2(channels,1, 17, 25, 0, 0)   
      elif model_tpye == 105:
            channels['skip_out_channels'] =    [48, 128, 256, 512, 512]
            channels['decoder_in_channels'] =  [48, 128, 256, 512, 512]
            channels['decoder_out_channels'] = [[96,96,96], [128,128,128],[256,256,256],[384,384,384],[512,512,512]]
            channels['aux_out_channels'] =  1
            temp = np.array(channels['decoder_out_channels'])
            channels['predict_in_channels'] = [96,96,96, temp.sum()]
            channels['aux_in_channels'] = [128,128,128,256,256,256,384,384,384,512,512,512]
            net = model8_3(channels,1, 6, 23, 4, 68)
      elif model_tpye == 107:
            channels['skip_out_channels'] =    [48, 128, 256, 512, 512]
            channels['decoder_in_channels'] =  [48, 128, 256, 512, 512]
            channels['decoder_out_channels'] = [[96,96,96], [128,128,128],[256,256,256],[384,384,384],[512,512,512]]
            channels['aux_out_channels'] =  1
            temp = np.array(channels['decoder_out_channels'])
            channels['predict_in_channels'] = [96,96,96]
            channels['aux_in_channels'] = [128,128,128,256,256,256,384,384,384,512,512,512]
            net = model8_3(channels,1, 6, 23, 4, 70)
      elif model_tpye == 108:
            channels['skip_out_channels'] =    [48, 128, 256, 512, 512]
            channels['decoder_in_channels'] =  [48, 128, 256, 512, 512]
            channels['decoder_out_channels'] = [[96,96,96], [128,128,128],[256,256,256],[384,384,384],[512,512,512]]
            channels['aux_out_channels'] =  1
            temp = np.array(channels['decoder_out_channels'])
            channels['predict_in_channels'] = [96,96,96]
            channels['aux_in_channels'] = [128,128,128,256,256,256,384,384,384,512,512,512]
            net = model8_3(channels,1, 6, 23, 4, 71) 
      elif model_tpye == 200:
            channels['skip_out_channels'] =    [48, 128, 256, 512, 512]
            channels['decoder_in_channels'] =  [48, 128, 256, 512, 512]
            channels['decoder_out_channels'] = [[64,64,64], [128,128,128],[256,256,256],[384,384,384],[512,512,512]]
            channels['aux_out_channels'] =  1
            temp = np.array(channels['decoder_out_channels'])
            channels['predict_in_channels'] = [64,64,64]
            channels['aux_in_channels'] = [128,128,128,256,256,256,384,384,384,512,512,512]
            net = model8_3(channels,1, 6, 23, 4, 73)
      return net 

def zero_one(img):
    mind = img.min()
    maxd = img.max()
    img = (img-mind) / (maxd-mind)
    return img
if __name__ == '__main__':
  device_num = 2
  last_ckpt = '/media/hpc/data/work/1/result/nju2000-4/au/model_tpye_108/1/ckpt/ckpt_epoch_273.00.pth'
  train_dataset = nda.read_RGBD(file_path = '/media/hpc/data/work/1/visual/viusal.npy',
                     transform=transforms.Compose([
                           nda.Resize(224,224),
                           nda.ToTensor_aux(), 
                           nda.zero_one_aux(),
                   ])
                          ) 
  train_data_loader = DataLoader(train_dataset, 1,shuffle=True, num_workers=1)
  print(len(train_data_loader))
  
  device = torch.device("cuda", device_num)
  torch.cuda.set_device(device_num)
  net = get_model(108)
  net.to(device)
  global_step, start_epoch = load_ckpt(net, None, last_ckpt, device) 
  weights = []
  with torch.no_grad():
      net.eval()
      for i_batch, sample_batched in enumerate(train_data_loader):
        rgb = sample_batched['rgb'].to(device)
        depth = sample_batched['depth'].to(device)
        gt = sample_batched['gt'].to(device)
        name = sample_batched['name']
#        out, rd, r, d = net(rgb, depth)
#        save_mask(out, rd, r, d, weight, name[0])
        fr, frd, fd = net(rgb, depth)
        save_feature_map4(name[0], fr[0], frd[0], fd[0])
#        save_feature_map3(out, au0_1, r_f, d_f,fuse, name[0], weight)
#        wrd, wr, wd = process_weight(weight)
##        weights.append([wrd, wr, wd])
#  weights=np.array(weights)
#  np.save('visual/weight97.npy', weights)        