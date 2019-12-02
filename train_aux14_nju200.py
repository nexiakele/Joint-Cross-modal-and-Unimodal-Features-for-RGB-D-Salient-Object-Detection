# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:46:16 2018

@author: Dell
"""
#############################################
import time
import numpy as np
#############################################
import torch
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms
import torch.nn.functional as F
#############################################
from tensorboardX import SummaryWriter
#############################################
import tools.Tools as tool
import tools.rgb_loader as nda
#############################################
import model as model
import Loss as ls
import argument2
#############################################
args = argument2.args  
args.name = 'au'
def trian():
###############################################################################
###############################################################################
########读取数据， 分别从验证集和训练集读取数据#####
    print('=>1：对数据没有进行正规化处理')
    train_dataset = nda.read_RGBD(file_path = args.train_file_path,
                                     transform=transforms.Compose([
                                                   nda.RandomCrop((224, 224)),
                                                   nda.RandomFlip(),
                                                   nda.RandomRot90(),
                                                   nda.RandomHSV((0.9,1.1),(0.9,1.1), (25,25), 0.25),
                                                   nda.ToTensor_aux6(), 
                                                   nda.zero_one_aux6(),
                                                   ]))  

########train_data_loader： 训练集数据############
    train_data_loader = DataLoader(train_dataset, args.batch_size,shuffle=True, num_workers=4)
###############################################################################
###############################################################################
####准备设备,当args.no_cuda == False 并且机器上存在GPU时，使用GPU进行训练网络
####device:默认的设备
    if  args.is_cuda and torch.cuda.is_available():
          device = torch.device("cuda", args.device_number)
          torch.cuda.set_device(args.device_number)
    else:
          device = torch.device("cpu")
    
    print('=>2: 使用的GPU为: ',  device)
####为CPU设置种子用于生成随机数，以使得结果是确定的 
    torch.manual_seed(args.seed)
###############################################################################
###############################################################################
####加载模型: model  准备Loss  准备优化器；optimizer######
    print('=>3: 训练的model_tpye为: ', args.model_type)
    net = model.get_model(args.model_type)
    net.to(device)
    criterion = ls.get_loss(args.loss_type,None)
    criterion.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum, 
                    weight_decay=args.weight_decay, nesterov = True)
    
    scheduler=StepLR(optimizer,step_size=args.lr_epoch_per_decay,gamma=args.lr_decay_rate) 
    
###############################################################################
###############################################################################
####加载模型: 根据设置载入已经训练好的模型##################
    if args.last_ckpt is not '':
          print('=>4: 载入训练好的模型的参数')
          global_step, args.start_epoch = tool.load_ckpt(net, optimizer, args.last_ckpt, device)
          summary_dir , ckpt_dir, time_dir = tool.get_dir_through_ckpt(args.last_ckpt, args)
          
          loss_report = np.load(summary_dir + '/loss.npy').tolist()
          val_record  = np.load(summary_dir + '/val_loss.npy').tolist()
          val_mse_record = np.load(summary_dir + '/val_mse_loss.npy').tolist()
          if  args.retrain_with_opt:
               summary_dir,ckpt_dir,time_dir=tool.make_dir(args)
    else:
##########创建目录#################
          print('=>4: 建立新的模型，创建目录文件')
          summary_dir,ckpt_dir,time_dir=tool.make_dir(args)
          print('==>文件名:', summary_dir,ckpt_dir,time_dir)
          loss_report = []
          val_record = []
          val_mse_record = []
##################################
    print('=>5: 优化器的参数:' ,optimizer)
    print('=>6: 网络的设置参数:')
    print(args.string)
    print('==================>开始训练<==================')
##################################
    global_step = 0
###############################################################################
##############################开始训练##########################################
    for epoch in range(int(args.start_epoch), args.epochs):
########根据需要更新学习率
        scheduler.step(max(0,(epoch-args.lr_epoch_decay)))
        ########保存网络参数和损失曲线
        if args.checkpoint and epoch % args.ckpt_epoch == 0:
            ####保存网络参数
            if epoch >= args.start_ckpt_epoch :
                  tool.save_ckpt(ckpt_dir, net, optimizer, global_step, epoch)
            ####保存网络损失曲线
            np.save(summary_dir + '/loss', loss_report) 
            np.save(summary_dir + '/val_loss', val_record)  
            np.save(summary_dir + '/val_mse_loss', val_mse_record)
###############################################################################    
#############################训练集#############################################
        net.train()
        train_total_loss = 0
        for i_batch, sample_batched in enumerate(train_data_loader):
            #### 取数据 ####
            rgb = sample_batched['rgb'].to(device)
            depth = sample_batched['depth'].to(device)
            gt = sample_batched['gt'].to(device)
            gt0 = sample_batched['gt0'].to(device)
            gt1 = sample_batched['gt1'].to(device)
            gt2 = sample_batched['gt2'].to(device)
            gt3 = sample_batched['gt3'].to(device)
            ###### 网络输出 #####
            out0,out1, out2,out3, out4 = net(rgb, depth)
            ###### 计算损失 #####
            loss, loss_record = criterion((out0,out1, out2,out3, out4), (gt, gt0, gt1, gt2, gt3))
            #### 梯度打印 ####
            if args.is_print_grad == True:
                  tool.print_grad(net.named_parameters(),args.grad_key_list)
            #### 梯度反向传播 ###
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            #### step信息输出 ###
            train_total_loss += loss.item()
            #### 数据打印 ######
            print('Epoch:', epoch ,
                  'step:' , i_batch, 
                  'lr:', optimizer.param_groups[0]['lr'],
                  'totall loss:', loss.item(),
                  'main loss:', loss_record[0], 
                  'au loss1:', loss_record[1],
                  'au loss2:', loss_record[2],
                  'au loss3:', loss_record[3],
                  'au loss4:', loss_record[4],)
            #### 数据存储 #####
        #### 数据存储 ####
        train_total_loss = train_total_loss / len(train_data_loader)
        loss_report.append(train_total_loss)
        #### 数据打印 ####
        print('loss at epoch:' + str(epoch) + ':', train_total_loss)
###############################################################################
###############################################################################
    ####数据存储###
    np.save(summary_dir + '/loss', loss_report) 
    np.save(summary_dir + '/val_loss', val_record)  
    np.save(summary_dir + '/val_mse_loss', val_mse_record)
    tool.save_ckpt(ckpt_dir, net, optimizer, global_step, epoch)
    
if __name__ == '__main__':
    trian()
