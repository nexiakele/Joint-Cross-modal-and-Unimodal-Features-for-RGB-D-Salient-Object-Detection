# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:30:25 2018

@author: Dell
"""

import os
import torch
import pandas as pd
from skimage import io, transform, color

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import random
import torch.nn.functional as F
from torch import nn

class read_image(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #print(file_path)
        self.file_path = file_path
        self.path = np.load(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        names = self.path[idx]
        rgb = io.imread(names[0]) 
        depth = io.imread(names[1])
        gt = io.imread(names[2])
        
        gt = color.rgb2grey(gt)
        #print(gt.shape)
        
        #show_all(rgb, depth, gt)
        #print(gt.shape)
        sample = {'rgb': rgb, 'depth': depth, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample

class read_RGBD(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, file_path,  transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #print(file_path)
        self.file_path = file_path
        self.path = np.load(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):

        names = self.path[idx]
        rgb = io.imread(names[0]) 
        depth = io.imread(names[1])
        gt = io.imread(names[2])
        

        
        #show_all(rgb, depth, gt)
        #print(gt.shape)
        sample = {'rgb': rgb, 'depth': depth, 'gt': gt}

        if self.transform:
            sample = self.transform(sample)

        return sample



        
def image_tf(image, target_width,target_height):
    image = image.resize((target_width,target_height), Image.BICUBIC)
    return image       
def show_data(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.show()
    
def show_all(rgb, depth, gt):
    show_data(rgb)
    show_data(depth)
    show_data(gt)        
class PreResize(object):
    """
    调整样本尺寸。使用sub-pixel进行上采样时，图像的尺寸必须为下采样的倍数。这里为16
    在测试时将图片重新调整到下采样的倍数防止运行错误。
    arg: 
        输入： sample中包含了 rgb, depth, gt图像，训练时统一为 292x224尺寸
              w, h 希望调整的尺寸， 若w,h 为0，则根据下采样的尺度自动调整图像的尺寸
        输出： 输出剪切后的rgb, depth, gt图像， 尺寸统一为 256x192
    """    
    
    auto_resize = False
    
    def __init__(self, w = 0, h = 0, scale = 16.0, order = [1,1,0]):
        if w == 0 or h == 0:
            self.auto_resize = True
        self.w = w
        self.h = h  
        self.scale = scale
        self.order =order
    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']

        if self.auto_resize :
            h, w = rgb.shape[:2]
            
            if w % self.scale == 0 and h % self.scale == 0:
                #print('尺寸正确')
                return sample
            else: 
                
                ws = int(w/self.scale)
                hs = int(h/self.scale)
                #print('调整尺寸为：', str(ws * 16), str(hs * 16))
                self.w = int(ws * self.scale)
                self.h = int(hs * self.scale)
            
        # Bi-linear       
        rgb = transform.resize(rgb, (self.h, self.w),order=self.order[0], 
                               mode='reflect', preserve_range=False, anti_aliasing=False)        
        
        depth = transform.resize(depth, (self.h, self.w),order=self.order[1], 
                                 mode='reflect', preserve_range=False, anti_aliasing=False)
        
        
        gt = transform.resize(gt, (self.h, self.w),order=self.order[2], 
                              mode='reflect', preserve_range=False, anti_aliasing=False)
         
        #show_all(rgb, depth, gt)
        return {'rgb': rgb, 'depth': depth, 'gt': gt}        
        
        
        
        
class RandomCrop(object):
    """
    样本增强-随机剪切
    arg: 
        输入： sample中包含了 rgb, depth, gt图像，训练时统一为 292x224尺寸
              ouput_size: 随机剪切后的尺寸
        输出： 输出剪切后的rgb, depth, gt图像， 尺寸统一为 256x192
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']

        h, w = rgb.shape[:2]
        #print(rgb.size)
        new_w, new_h = self.output_size
        
        if w <= new_w or h <= new_h:
            #print('剪切尺寸大于原始尺寸')
            return {'rgb': rgb, 'depth': depth, 'gt': gt}
      
        i = random.randint(0, h - new_h)
        j = random.randint(0, w - new_w)
        
        rgb = rgb[i:i + new_h, j:j + new_w, :]
        depth = depth[i:i + new_h, j:j + new_w, :]
        gt = gt[i:i + new_h, j:j + new_w]
        #show_all(rgb, depth, gt)
        return {'rgb': rgb, 'depth': depth, 'gt': gt}        
        
        
class RandomFlip(object):
    """
    样本增强-随机水平和垂直翻转，默认概率为0.5
    arg: 
        输入： sample中包含了 rgb, depth, gt图像 
               
        输出： 输出翻转后的的rgb, depth, gt图像 
    """    
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']
        if random.random() < self.p:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy() 
            gt = np.fliplr(gt).copy() 
            
        if random.random() < self.p:
            rgb = np.flipud(rgb).copy()
            depth = np.flipud(depth).copy()
            gt = np.flipud(gt).copy()     
        
        #show_all(rgb, depth, gt)
        return {'rgb': rgb, 'depth': depth, 'gt': gt}         
        
        
        
        
class RandomRot90(object):
    """
    样本增强-随机旋转90度
    arg: 
        输入： sample中包含了 rgb, depth, gt图像 
        输出： 输出旋转90度的rgb, depth, gt图像 
    """  
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, sample):
        """

        """
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']
        rd = random.random()
        if  rd < self.p :
            rgb = transform.rotate(rgb,90) #  rgb.rotate(90)
            depth = transform.rotate(depth,90)
            gt = transform.rotate(gt,90)
        elif rd < 2 * self.p :
            rgb = transform.rotate(rgb,270) #  rgb.rotate(90)
            depth = transform.rotate(depth,270)
            gt = transform.rotate(gt,270)
        #show_all(rgb, depth, gt)    
        return {'rgb': rgb, 'depth': depth, 'gt': gt}          
        

class ToTensor(object):

    def __call__(self, sample,  to_byte = False):
        rgb, depth, gt = sample['rgb'], sample['depth'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #np.expand_dims(gt, 0)
#        rgb = rgb.transpose((2, 0, 1))
#        depth = depth.transpose((2, 0, 1))
#        gt = np.expand_dims(gt, 0).astype(np.float)
        rgb =  torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        depth =  torch.from_numpy(depth.transpose((2, 0, 1))).float()
        gt =  torch.from_numpy(np.expand_dims(gt, 0)).float()  
        gt = gt / 255.0
        if to_byte:
              rgb = rgb / 255.0
              depth = depth / 255.0
      
        return {'rgb': rgb, 'depth': depth, 'gt': gt}    
    
def test():
    
#    plt.ion()   # interactive mode
#    fig = plt.figure()

    train_file_path = './sub-pixel/path/train_path.npy'
    test_file_path = './datasave/test_path.npy'
    val_file_path = './datasave/val_path.npy'
    
    img = read_image(train_file_path)
    
    for idx in range (20):
        images = img.__getitem__(idx)
        gt= images['gt']
        print(gt)
        plt.imshow(gt, cmap='gray')
        plt.show()
        print(idx)
        break
#    transformed_dataset = read_image(file_path = train_file_path,
#                                     transform=transforms.Compose([
#                                       PreResize(292,224),
#                                       RandomCrop((256, 192)),
#                                       RandomRot90(),
#                                       RandomFlip(),
#                                       ToTensor()
#                                   ]))
#    dataloader = DataLoader(transformed_dataset, batch_size=4,
#                        shuffle=True, num_workers=4)
#
#    print(len(dataloader))
#    for i_batch, sample_batched in enumerate(dataloader):
#        print(i_batch, sample_batched['rgb'].size(),
#              sample_batched['depth'].size(),
#              sample_batched['gt'].size(), sample_batched['gt'].dtype)
#        #show_tensor_batch(sample_batched)
#        # observe 4th batch and stop.
#        gt = sample_batched['gt'][0]
#        print(gt.shape)
##        plt.imshow(gt)
##        plt.show()
#        if i_batch == 3:
##            plt.figure()
##            show_batch(sample_batched)
##            plt.axis('off')
##            plt.ioff()
##            plt.show()
#            break 
    

    #grid2 = utils.make_grid(depth_batch)
    #plt.imshow(grid2.numpy().transpose((1, 2, 0)))
    #grid3 = utils.make_grid(gt_batch)
    #plt.imshow(grid3.numpy().transpose((1, 2, 0)))
 
#    rm = read_image(train_file_path)
#    
#    sample =rm.__getitem__(3)
#    
#
#    pr = PreResize(292,224)
#    sample = pr.__call__(sample)
##    rc = RandomCrop((256, 192))
##    sample = rc.__call__(sample)
#    rt90 =RandomRot90()
#    rt90.__call__(sample)
##    
#    rf = RandomFlip()
#    rf.__call__(sample)
#    show_all(sample['rgb'], sample['depth'],sample['gt'])    
#    print(sample['gt'])
#    TT = ToTensor()
#    sample = TT.__call__(sample)
#    print(sample['rgb'].shape)
#    print(sample['gt'].shape)
# Helper function to show a batch
    
    
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    rgb_batch, depth_batch, gt_batch = \
            sample_batched['rgb'], sample_batched['depth'],sample_batched['gt']
    batch_size = len(rgb_batch)
    im_size = rgb_batch.size(2)

    grid = utils.make_grid(rgb_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    
def show_out_gt(output):
    out = output.numpy()
    out = out[1,:,:,:]
    out = transpose((1, 2, 0))
    plt.imshow(out)
    plt.show()

def im_show(img):
      plt.imshow(img)
      plt.show()      
def test2():
      re = read_RGBD('../dataset/nju2000/train_path.npy')
      sample = re.__getitem__(1)
      gt = sample['gt']
      im_show(sample['gt'])
      im_show(sample['rgb'])
      im_show(sample['depth'])
if __name__ == '__main__':
    test2()
