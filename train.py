# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:46:16 2018

@author: Dell
"""
import time
import torch
from torch.utils.data import DataLoader
import torch.optim
from tensorboardX import SummaryWriter
import torch.nn as nn
import tools.Tools as tool
import numpy as np
import torch.nn.functional as F
import tools.rgb_loader as nda
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

import model as model
import argument
args = argument.args 
args.name = 'base'
args.string = argument.get_argString(args)
def trian():
###############################################################################
###############################################################################
########读取数据， 分别从验证集和训练集读取数据#####
    train_dataset = nda.read_RGBD(file_path = args.train_file_path,
                   transform=transforms.Compose([
                           nda.RandomCrop((320, 192)),
                           nda.RandomFlip(),
                           nda.RandomHSV((0.9,1.1),(0.9,1.1), (25,25), 0.25),
                           nda.ToTensor(), 
                           nda.zero_one(),
                           ]))  
    val_dataset = nda.read_RGBD(file_path = args.val_file_path,
                               transform=transforms.Compose([
                                 nda.RandomRot90(),
                                 nda.RandomFlip(),
                                 nda.RandomHSV((0.9, 1.1),(0.9, 1.1),(25, 25),0.25),
                                 nda.ToTensor(), 
                                 nda.zero_one()
                             ]))   
########train_data_loader： 训练集数据############
    train_data_loader = DataLoader(train_dataset, args.batch_size,shuffle=True, num_workers=4)
########val_data_loader： 验证集数据##############
    val_data_loader = DataLoader(val_dataset, 2,shuffle=False, num_workers=2)
    print(len(train_data_loader))
###############################################################################
###############################################################################
####准备设备,当args.no_cuda == False 并且机器上存在GPU时，使用GPU进行训练网络
####device:默认的设备
    if  args.is_cuda and torch.cuda.is_available():
          device = torch.device("cuda", args.device_number)
          torch.cuda.set_device(args.device_number)
    else:
          device = torch.device("cpu")
    print('device is :',  device)
####为CPU设置种子用于生成随机数，以使得结果是确定的 
    torch.manual_seed(args.seed)
###############################################################################
###############################################################################
####加载模型: model  准备Loss  准备优化器；optimizer######
    print('model_tpye:', args.model_type)
    net = model.get_model(args.model_type)  
    net.to(device)
    net.apply(tool.weights_init)
    criterion = nn.BCELoss()
    criterion.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum, 
                                weight_decay=args.weight_decay, nesterov=False)
    
    scheduler=StepLR(optimizer,step_size=args.lr_epoch_per_decay,gamma=args.lr_decay_rate) 
    print('optimizer:' ,optimizer)
###############################################################################
###############################################################################
####加载模型: 根据设置载入已经训练好的模型##################
    if args.last_ckpt is not '':
          print('loading ckpt and making dir')
          global_step, args.start_epoch = tool.load_ckpt(net, optimizer, args.last_ckpt, device)
          summary_dir , ckpt_dir, time_dir = tool.get_dir_through_ckpt(args.last_ckpt)
          loss_report = np.load(time_dir + '/loss.npy') 
          val_record  = np.load(time_dir + '/val_loss.npy')  
          val_mse_record = np.load(time_dir + '/val_mse_loss.npy')
    else:
##########创建目录#################
          print('making dir ...')
          summary_dir,ckpt_dir,time_dir=tool.make_dir(args.name,args.model_type, args.string)
          loss_report = []
          val_record = []
          val_mse_record = []
##################################
    print('network setup:')
    print(args.string)
    print('start train')
##################################
    global_step = 0
    val_global_step = 0
    writer = SummaryWriter(log_dir=summary_dir)
###############################################################################
##############################开始训练##########################################
    for epoch in range(int(args.start_epoch), args.epochs):
########根据需要更新学习率
        scheduler.step(max(0,(epoch-args.lr_epoch_decay)))
        if args.checkpoint and epoch % args.ckpt_epoch == 0 and epoch != args.start_epoch:
            tool.save_ckpt(ckpt_dir, net, optimizer, global_step, epoch)
            np.save(time_dir + '/loss', loss_report) 
            np.save(time_dir + '/val_loss', val_record)  
            np.save(time_dir + '/val_mse_loss', val_mse_record)
            print('save success')
###############################################################################
################################# 验证集 #######################################
        if args.is_val and epoch % args.val_freq == 0 and epoch !=0 :
########### 取消梯度 #################
            with torch.no_grad():
################# 取消BN #############
                  net.eval()
                  val_total_loss = 0
                  totall_mse = 0
                  for i_batch, sample_batched in enumerate(val_data_loader):
####################### 加载数据 #######################
                        rgb = sample_batched['rgb'].to(device)
                        depth = sample_batched['depth'].to(device)
                        gt = sample_batched['gt'].to(device)
####################### 网络输出 #######################
                        output = net(rgb, depth)
####################### 计算损失 #######################
                        loss = criterion(output, gt)
                        mse = F.l1_loss(output, gt)
####################### 损失统计 #########################
                        val_total_loss += loss.item()
                        totall_mse = totall_mse + mse.item()
                        val_global_step =  val_global_step + 1
################## 计算平均损失 #######################
                  totall_mse = totall_mse/len(val_data_loader)
                  val_total_loss = val_total_loss/len(val_data_loader)
##################数据存储########################
                  writer.add_scalar('val_mse_record', totall_mse, epoch)
                  writer.add_scalar('val_total_loss', val_total_loss, epoch)
                  val_mse_record.append(totall_mse)     
                  val_record.append(val_total_loss)   
##################损失打印########################
                  print('val_total_loss: ', val_total_loss)
                  print('val_mse_record: ', totall_mse)
###############################################################################    
############################ 训练集 ############################################
        net.train()
        train_total_loss = 0
        for i_batch, sample_batched in enumerate(train_data_loader):
############ 取数据 ###########################
            rgb = sample_batched['rgb'].to(device)
            depth = sample_batched['depth'].to(device)
            gt = sample_batched['gt'].to(device)
########### 网络输出 ###########################
            output = net(rgb, depth)
########### 损失输出 ###########################
            loss = criterion(output, gt)
########### 梯度打印 ###########################
            if args.is_print_grad == True:
                  tool.print_grad(net.named_parameters(),args.grad_key_list)
############ 梯度反向传播 ######################
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
########### 数据打印 ##########################
            train_total_loss += loss.item()
            time_str = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime())
            print(time_str +' Epoch:', epoch , 'step:'+ str(i_batch) + ' loss:', 
                  loss.item() , '  lr:', optimizer.param_groups[0]['lr'])
########### 数据存储 ################################
            writer.add_scalar('train_loss', loss, global_step)
######## 数据存储 ################################
        train_total_loss = train_total_loss / len(train_data_loader)
        loss_report.append(train_total_loss)
        writer.add_scalar('train_loss_epoch', train_total_loss, epoch)
####### 数据打印 ##########################
        print('loss at epoch:' + str(epoch) + ':', train_total_loss)
#######################################################################
#######################################################################
#### 数据存储 ###################################
    np.save(time_dir + '/loss', loss_report) 
    np.save(time_dir + '/val_loss', val_record)  
    np.save(time_dir + '/val_mse_loss', val_mse_record)
    tool.save_ckpt(ckpt_dir, net, optimizer, global_step, epoch)

    writer.close()
if __name__ == '__main__':
    trian()
