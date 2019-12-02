# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:17:54 2018

@author: Dell
"""
import torch
from torch.utils.data import DataLoader
import torch.optim
from torch import nn
from tools.Tools import load_ckpt,  make_dir, save_d
import tools.rgb_loader as nda
import torchvision.transforms as transforms

import model as model
import argument
args = argument.args
args.name = 'infenc_base'
args.string = argument.get_argString(args)
def infence():
    """ 
    读取数据， 分别从验证集和训练集读取数据
    train_data_loader： 训练集数据
    val_data_loader： 验证集数据
    """
    test_dataset = nda.read_RGBD(file_path = args.test_file_path,
                               transform=transforms.Compose([
                                 nda.ToTensor(), 
                                 nda.zero_one()
                             ]))
    test_data_loader = DataLoader(test_dataset, 1,shuffle=False, num_workers=2)

    """ 
     准备设备： 
     args.no_cuda == False 并且机器上存在GPU时，使用GPU进行训练网络
     device:默认的设备
    """    
    if  args.is_cuda and torch.cuda.is_available():
          device = torch.device("cuda", args.device_number)
          torch.cuda.set_device(args.device_number)
    else:
          device = torch.device("cpu")
    print('device is :',  device)  

    print('model_tpye:', args.model_type)
    net = model.get_model(args.model_type)    
    net.to(device)  
    criterion = nn.BCELoss()
    criterion.to(device)
    l1_loss = nn.L1Loss()
    l1_loss.to(device)
    """ 
    加载模型: 根据设置载入已经训练好的模型
    """        
    if args.last_ckpt is not '':
        global_step, args.start_epoch = load_ckpt(net, None, args.last_ckpt, device)

    print('making dir ...')
    summary_dir , ckpt_dir, time_dir = make_dir(args.name, args.model_type,  args.string)
    
    print('start test')
    mean_cross_loss = 0
    mean_l1_loss = 0
    with torch.no_grad():
          net.eval()
          for i_batch, sample_batched in enumerate(test_data_loader):
            rgb = sample_batched['rgb'].to(device)
            depth = sample_batched['depth'].to(device)
            gt = sample_batched['gt'].to(device)
            name = sample_batched['name']
            output = net(rgb, depth)
            loss = criterion(output, gt)
            l1 = l1_loss(output, gt)
            print(name[0], ' done, cross loss: ', loss.item(), ' L1 loss:',l1.item() )
            save_d( output  , summary_dir + '/' + name[0] + '.png')
            mean_cross_loss = mean_cross_loss + loss.item()
            mean_l1_loss  =   mean_l1_loss + l1.item()
            
          mean_cross_loss = mean_cross_loss / len(test_data_loader)
          mean_l1_loss    = mean_l1_loss / len(test_data_loader)
          print('cross_loss: ' ,mean_cross_loss, '\t mean_l1_loss:', mean_l1_loss)

if __name__ == '__main__':
    infence()