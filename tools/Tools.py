# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 16:21:35 2018

@author: Dell
"""
import numpy as np
from torch import nn
import torch
import os
from torch.nn import init
import matplotlib.pyplot as plt
import time
from skimage import io
############################################################
############################################################
def show_rgb(img_tensor):
    out = img_tensor.detach().numpy()
    out = out[0]
    out = out.transpose((1, 2, 0))
    plt.imshow(out)
    plt.show()
    
def save_d(img_tensor, dir_path):
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()
    out = out[0]
    out =  out[0]
    out[out > 1] = 1
    out[out<-1] = -1
    io.imsave(dir_path,out)
def show_d(img_tensor):
    out = img_tensor.detach().numpy()
    out = out[0]
    out =  out[0]
    
    plt.imshow(out)
    plt.show()
############################################################
#####################备份与恢复##############################
#########备份########
def save_ckpt(ckpt_dir, model, optimizer, global_step, epoch):
    # usually this happens only on the start of a epoch
    state = {
        'global_step': global_step,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    ckpt_model_filename = "ckpt_epoch_{:0.2f}.pth".format(epoch)
    path = ckpt_dir + '/' + ckpt_model_filename
    time.sleep(5)
    torch.save(state, path)
    #创建识别文件
    check_path = ckpt_dir + '/0-0-0-0-0-0-0-0-0.txt'
    if not os.path.exists(check_path):
          with open(check_path, 'w') as f:
                f.write('check')
    print('===> {:>2} has been successfully saved'.format(path))
#########恢复########
def load_ckpt(model, optimizer, model_file, device):
    if os.path.isfile(model_file):
        print("===> loading checkpoint '{}'".format(model_file))
        if device.type == 'cuda':
            checkpoint = torch.load(model_file)
        else:
            checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("===> loaded checkpoint '{}' (epoch {})"
              .format(model_file, checkpoint['epoch']))
        step = checkpoint['global_step']
        epoch = checkpoint['epoch']
        return step, epoch
    else:
        print("===> no checkpoint found at '{}'".format(model_file))
        os._exit(0)
############################################################
#################创建文件目录#################################
def make_dir(args):
    #创建类型目录， name =  aux or base
    father_dir =  './result/' + str(args.dataset_name)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    father_dir =  father_dir + '/' + str(args.name)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建网络类型目录
    father_dir =   father_dir + '/model_tpye_'+ str(args.model_type)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建时间目录
    dirs = os.listdir( father_dir )
    trian_times = max(len(dirs), 1)
    time_dir = father_dir + '/' + str(trian_times)
    if os.path.exists(time_dir+'/ckpt/0-0-0-0-0-0-0-0-0.txt'):
          trian_times = len(dirs) + 1
    time_dir = father_dir + '/' + str(trian_times)
    if not os.path.exists(time_dir) :
        os.mkdir(time_dir)
    summary_dir = time_dir + '/summary_' + str(args.model_type)
    if not os.path.exists(summary_dir):
        os.mkdir(summary_dir)
    ckpt_dir =  time_dir + '/ckpt'
    if not os.path.exists(ckpt_dir):
              os.mkdir(ckpt_dir)
    with open(time_dir + '/remarks.txt' , 'w') as f:
          f.write(args.string)
    return summary_dir, ckpt_dir, time_dir
############################################################
def make_infence_dir(args, is_bad=False):
    #创建类型目录， name =  aux or base
    time_path = args.last_ckpt.split('/')[-3]
    father_dir =  './result/' + str(args.dataset_name)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    father_dir =  father_dir + '/' + str(args.name)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建网络类型目录
    father_dir =   father_dir + '/model_tpye_'+ str(args.model_type)
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    #创建时间目录
    father_dir =  father_dir + '/' +time_path
    if not os.path.exists(father_dir):
          os.mkdir(father_dir)
    time_dir = father_dir + '/'+str(args.model_type)+'_'+time_path +'_epochs_' + str(args.start_epoch)
    if not os.path.exists(time_dir) :
        os.mkdir(time_dir)
    ##########################################
    if is_bad:
          bad_result_dir = time_dir + '/bad_result'
          if not os.path.exists(bad_result_dir) :
              os.mkdir(bad_result_dir)
          bad_output = bad_result_dir + '/output'
          if not os.path.exists(bad_output) :
              os.mkdir(bad_output)
          bad_gt = bad_result_dir + '/gt'
          if not os.path.exists(bad_gt) :
              os.mkdir(bad_gt)
          bad_depth = bad_result_dir + '/depth'
          if not os.path.exists(bad_depth) :
              os.mkdir(bad_depth)
          bad_rgb = bad_result_dir + '/rgb'
          if not os.path.exists(bad_rgb) :
              os.mkdir(bad_rgb)
    else:
           bad_result_dir=None
    return time_dir, bad_result_dir
##############得到ckpt目录###########################
def get_dir_through_ckpt(ckpt_path,args):
      ckpt_dir = os.path.dirname(ckpt_path)
      time_dir = os.path.dirname(ckpt_dir)
      summary_dir =  time_dir + '/summary_' + str(args.model_type)
      return summary_dir, ckpt_dir, time_dir 

############################################################
###################网络权重初始化#############################
#######################################
# weight init
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        if m.bias is not None:
              init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)    
#######################################
def xavier(param):
    init.xavier_uniform_(param)
#######################################
def Gassion(param):
    init.normal_(param, mean=0, std=1) 
############################################################
#####################打印梯度################################
def print_grad(parameters, keys):
      for k, v in parameters:
            if k in keys:
                  print(k, '  grad is :')
                  h = v.register_hook(lambda grad: print(grad))
                  print('------------------------------------')
############################################################
############################################################
def save_rgb_batch(img_tensor, dir_path,name):
    p=lambda out,i,name:io.imsave(name,out[i].transpose((1, 2, 0)))
    out = img_tensor.detach().numpy()
    l   = len(name)
    for i in range(l):
          p(out, i,dir_path+name[i]+'.jpg')
def save_d_batch(img_tensor, dir_path,name):
    p=lambda out,i,j,name:io.imsave(name,out[i][j])
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()
    
    l   = len(name)
    for i in range(l):
          p(out, i,0,dir_path+name[i]+'.png')
          
def save_all(sample):
      dir_name = {
            'rgb' : './save_test/rgb/',
            'depth': './save_test/depth/',
            'gt' : './save_test/gt/',
            'gt0' : './save_test/gt0/',
            'gt1' : './save_test/gt1/',
            'gt2' : './save_test/gt2/',
            }
      rgb = sample['rgb']
      depth = sample['depth']
      gt = sample['gt']
      gt0 = sample['gt0']
      gt1 = sample['gt1']
      gt2 = sample['gt2']
      name = sample['name']
      
      save_rgb_batch(rgb,dir_name['rgb'], name )
      save_d_batch(gt,dir_name['gt'], name )
      save_d_batch(depth,dir_name['depth'], name )
      save_d_batch(gt0,dir_name['gt0'], name )
      save_d_batch(gt1,dir_name['gt1'], name )
      save_d_batch(gt2,dir_name['gt2'], name )
def is_cpkt(epochs, start_epoch):
      if epochs < start_epoch :
            if epochs == 0:
              return False
            return (epochs % 20 == 0)
      else:
            return True
############################################################
############################################################
if __name__ == '__main__':
    string = '/media/hpc/data/work/work/work/result/nju2000/au/model_tpye_14/2019-01-02-09-09/ckpt/ckpt_epoch_249.00.pth'
#    print(string.split('/')[-3])[[1,0,0],[1,0,0],[0,0,0]],[[1,0,0],[1,0,0],[1,0,0]]
    
#      img = np.array([[[[1,0,0],[0,0,0],[0,0,0]]],
#                      [[[1,0,0],[1,0,0],[0,0,0]]],
#                      [[[1,0,0],[1,0,0],[1,0,0]]],
#                      [[[1,1,1],[1,1,1],[1,1,1]]]])
#      img = torch.from_numpy(img).float()
##      print(img[3,:,:,:].sum())
##      print(img.sum(dim=0))
##      print(img.sum(dim=1))
##      print(img.sum(dim=2))
##      print(img.sum(dim=3))