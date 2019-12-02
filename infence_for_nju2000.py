# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 22:17:54 2018

@author: Dell
"""
import torch
from torch.utils.data import DataLoader
import torch.optim
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage import io
from tools.Tools import load_ckpt, make_infence_dir
import tools.rgb_loader as nda
import matplotlib.image  as mpimg
import model as model
import argument2

args = argument2.args  
args.name = 'infence_aux'
def takeSecond(elem):
    return elem[1]
def save_bad_result(output,rgb,depth,gt,dir_path,name,loss):
      rgb_dir = dir_path + '/rgb/' + name  + '.jpg'
      gt_dir = dir_path + '/gt/' + name  + '.png'
      depth_dir = dir_path + '/depth/' + name  + '.jpg'
      output_dir = dir_path + '/output/' + name  + '_'+ str(loss) + '.png'
      output = output.detach().squeeze().cpu().numpy()
      rgb = rgb.detach().squeeze().cpu().numpy().transpose((1, 2, 0))
      depth = depth.detach().squeeze().cpu().numpy()
      gt = gt.detach().squeeze().cpu().numpy()
      mpimg.imsave(rgb_dir ,rgb, cmap='gray')
#      mpimg.imsave(depth_dir ,depth)
      mpimg.imsave(gt_dir ,gt, cmap='gray')
      mpimg.imsave(output_dir ,output, cmap='gray')
def save_d(img_tensor, dir_path):
    out = img_tensor.cpu().clone()
    out = out.detach().numpy()
    out = out[0]
    out =  out[0]
    out[out > 1] = 1
    out[out<-1] = -1
    mpimg.imsave(dir_path,out, cmap='gray')
def save_bad_result3(output, a1, a2, a3, a4,dir_path,name):
      output_dir = dir_path + '/output/' + name  + '_0' + '.png'
      output_dir1 = dir_path + '/output/' + name  + '_1' + '.png'
      output_dir2 = dir_path + '/output/' + name  + '_2' + '.png'
      output_dir3 = dir_path + '/output/' + name  + '_3' + '.png'
      output_dir4 = dir_path + '/output/' + name  + '_4' + '.png'
      output = output.detach().squeeze().cpu().numpy()
      a1 = a1.detach().squeeze().cpu().numpy()
      a2 = a2.detach().squeeze().cpu().numpy()
      a3 = a3.detach().squeeze().cpu().numpy()
      a4 = a4.detach().squeeze().cpu().numpy()
      mpimg.imsave(output_dir ,output, cmap='gray')
      mpimg.imsave(output_dir1 ,a1, cmap='gray')
      mpimg.imsave(output_dir2 ,a2, cmap='gray')
      mpimg.imsave(output_dir3 ,a3, cmap='gray')
      mpimg.imsave(output_dir4 ,a4, cmap='gray')
def save_bad_result2(output,rgb,depth,gt,dir_path,name,loss):
      gt_dir = dir_path + '/gt/' + name  + '.png'
      output_dir = dir_path + '/output/' + name  + '_'+ str(loss) + '.png'
      output = output.detach().squeeze().cpu().numpy()
      gt = gt.detach().squeeze().cpu().numpy()
      mpimg.imsave(gt_dir ,gt, cmap='gray')
      mpimg.imsave(output_dir ,output, cmap='gray')

def infence():
    """ 
    读取数据， 分别从验证集和训练集读取数据
    train_data_loader： 训练集数据
    val_data_loader： 验证集数据
    """
    if not args.is_per_img_norm :
          test_dataset = nda.read_RGBD(file_path = args.test_file_path,
                         transform=transforms.Compose([
                           nda.Resize(224,224),
                           nda.ToTensor_aux6(), 
                           nda.zero_one_aux6(),
                       ]))
          test_data_loader = DataLoader(test_dataset, 1,shuffle=False, num_workers=2)
    else:
          test_dataset = nda.read_RGBD(file_path = args.test_file_path,
                         transform=transforms.Compose([
                           nda.Resize(224,224),
                           nda.ToTensor_aux(), 
                           nda.zero_one_aux(),
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
    """ 
    加载模型: 根据设置载入已经训练好的模型
    """        
    if args.last_ckpt is not '':
        global_step, args.start_epoch = load_ckpt(net, None, args.last_ckpt, device)

    print('making dir ...')
    result_dir, bad_result_dir  = make_infence_dir(args)
    print('result dir is : ', result_dir)
    print('start test')
    
    bad_result = []
    with torch.no_grad():
          net.eval()
          val_total_loss =0
          totall_mse = 0
          for i_batch, sample_batched in enumerate(test_data_loader):
            rgb = sample_batched['rgb'].to(device)
            depth = sample_batched['depth'].to(device)
            gt = sample_batched['gt'].to(device)
            name = sample_batched['name']
#            if args.model_type < 41:
#                  output, a1, a2, a3 = net(rgb, depth)
#            else:
            out= net(rgb, depth)
            output = out[0][0] 
            loss = criterion(output, gt)
            
            mse = F.l1_loss(output, gt)
            
            totall_mse = totall_mse + mse.item()
            val_total_loss += loss.item()

            print(name[0], ' done, cross loss: ', loss.item(), '  l1 loss:', mse.item())
            save_d( output  , result_dir + '/' + name[0] + '.png')
            ########保存不好的结果
            if mse.item() > 0.1:
                if not args.is_per_img_norm :
                  save_bad_result(output, rgb,depth,gt,bad_result_dir,name[0],mse.item())
                else:
                  save_bad_result2(output, rgb,depth,gt,bad_result_dir,name[0],mse.item())
                bad_result.append((name[0], mse.item()))
#          save_bad_result3(output, a1, a2, a3, a4,bad_result_dir,name[0])
          totall_mse = totall_mse/len(test_data_loader)
          val_total_loss = val_total_loss/len(test_data_loader)
          print('cross loss :', val_total_loss, 'l1 loss :',  totall_mse)
          file=open(bad_result_dir + '/bad_result.txt','w')  
          bad_result.sort(key=takeSecond)
          file.write(str(bad_result));  
          file.close()
          

 

if __name__ == '__main__':
    infence()