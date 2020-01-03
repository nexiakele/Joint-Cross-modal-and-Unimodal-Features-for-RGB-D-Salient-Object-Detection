# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 22:36:21 2018

@author: Dell
"""


import torch
from skimage import io, transform, color

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import random
import torch.nn.functional as F
from torch import nn
import os
###############################################################################
###############################################################################
###############################################################################
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
        self.file_path = file_path
        self.path = np.loadtxt(file_path+'test.txt', dtype=str)
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
          
        names = self.path[idx]
        
        rgb = io.imread(self.file_path + names[0]) 
        depth = io.imread(self.file_path +names[1])
        gt = io.imread(self.file_path +names[2])
        name = names[3]
        if len(rgb.shape) == 2:
              rgb = color.gray2rgb(rgb)

        sample = {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
###############################################################################
###############################################################################
###############################################################################
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
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        h, w = rgb.shape[:2]
        new_w, new_h = self.output_size
        if w <= new_w or h <= new_h:
            #print('剪切尺寸大于原始尺寸')
            return {'rgb': rgb, 'depth': depth, 'gt': gt}
        
        i = random.randint(0, h - new_h)
        j = random.randint(0, w - new_w)
        
        rgb = rgb[i:i + new_h, j:j + new_w, :]
        depth = depth[i:i + new_h, j:j + new_w]
        gt = gt[i:i + new_h, j:j + new_w]
        #show_all(rgb, depth, gt)
        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}         
###############################################################################
###############################################################################
###############################################################################       
class RandomScale(object):
    def __init__(self, scale):
        self.scale_low = min(scale)
        self.scale_high = max(scale)
    def __call__(self, sample):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']

        target_scale = random.uniform(self.scale_low, self.scale_high)
        # (H, W, C)
        target_height = int(round(target_scale * rgb.shape[0]))
        target_width = int(round(target_scale * rgb.shape[1]))
        # Bi-linear
        rgb = transform.resize(rgb, (target_height, target_width),
                                         order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = transform.resize(depth, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)
        gt = transform.resize(gt, (target_height, target_width),
                                         order=0, mode='reflect', preserve_range=True)

        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}
###############################################################################
###############################################################################
###############################################################################       
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
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        if random.random() < self.p:
            rgb = np.fliplr(rgb).copy()
            depth = np.fliplr(depth).copy() 
            gt = np.fliplr(gt).copy() 
            
        if random.random() < self.p:
            rgb = np.flipud(rgb).copy()
            depth = np.flipud(depth).copy()
            gt = np.flipud(gt).copy()     
        
        #show_all(rgb, depth, gt)
        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}        
###############################################################################
###############################################################################
############################################################################### 
class Resize(object):
    """
    样本增强-随机水平和垂直翻转，默认概率为0.5
    arg: 
        输入： sample中包含了 rgb, depth, gt图像 
               
        输出： 输出翻转后的的rgb, depth, gt图像 
    """    
    def __init__(self, w = 320, h = 192):
        self.w = w
        self.h = h
    def __call__(self, sample):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        # Bi-linear
        rgb = transform.resize(rgb, 
                               (self.h, self.w),
                               order=1, 
                               mode='reflect', 
                               preserve_range=True,
                               anti_aliasing=False)
        # Nearest-neighbor
        depth = transform.resize(depth, 
                                 (self.h, self.w),
                                 order=0, 
                                 mode='reflect', 
                                 preserve_range=True,
                                 anti_aliasing=False)
        gt = transform.resize(gt, 
                              (self.h, self.w),
                              order=0, 
                              mode='reflect', 
                              preserve_range=True,
                              anti_aliasing=False)
        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}
###############################################################################
###############################################################################
###############################################################################
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
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        rd = random.random()
        if  rd < self.p :
            rgb = transform.rotate(rgb, 90, resize = False, order = 1, mode = 'reflect',
                                   clip = True, preserve_range = True)
            
            depth = transform.rotate(depth , 90, resize = False, order = 0, mode = 'reflect',
                                   clip = True, preserve_range = True)
            
            gt = transform.rotate(gt , 90, resize = False, order = 0, mode = 'reflect',
                                   clip = True, preserve_range = True)
        elif rd < 2 * self.p :
            rgb = transform.rotate(rgb, 270, resize = False, order = 1, mode = 'reflect',
                                   clip = True, preserve_range = True)
            depth = transform.rotate(depth , 270, resize = False, order = 0, mode = 'reflect',
                                   clip = True, preserve_range = True)
            gt = transform.rotate(gt , 270, resize = False, order = 0, mode = 'reflect',
                                   clip = True, preserve_range = True)
            
        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}
###############################################################################
###############################################################################
###############################################################################
class RandomGray(object):
    """
    样本增强-随机灰度
    arg: 
        输入： sample中包含了 rgb, depth, gt图像 
        输出： 输出 rgb, depth, gt图像 
    """  
    def __init__(self, p=0.05):
        self.p = p

    def __call__(self, sample):
        """

        """
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        rd = random.random()
        if  rd < self.p :
            rgb = color.rgb2gray(rgb)
            rgb = color.gray2rgb(rgb) 
        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}
###############################################################################
###############################################################################
###############################################################################  
class RandomHSV(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range, p=0.25):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.p = p
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
    def __call__(self, sample):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        rd = random.random()
        if  rd < self.p :
              rgb = self.randomhsv(rgb)
        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}
    def randomhsv(self, img):
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return img_new
###############################################################################
###############################################################################
############################################################################### 
class ToTensor(object):

    def __call__(self, sample,  to_byte = False, is_gt_to_byte = False):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']

        rgb =  torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        depth =  torch.from_numpy(np.expand_dims(depth, 0)).float()
        gt =  torch.from_numpy(np.expand_dims(gt, 0)).float()
        return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name} 
###############################################################################
###############################################################################
###############################################################################
class zero_one(object):
    def __call__(self, sample):
      rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']   
      rgb = rgb / 255.0
      depth = depth / 255.0
      gt = gt / 255.0
      return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}  
###############################################################################
###############################################################################
###############################################################################
class per_image_standardization(object):
      def __call__(self, sample):
            rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
            rgb = self.image_standardization(rgb)
            depth = self.image_standardization(depth)
            
            return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}
      
      def image_standardization(self, image_tensor):
            std = torch.std(image_tensor)
            mean = torch.mean(image_tensor)
            if std != 0:
                  image_tensor = (image_tensor - mean) / std
            return image_tensor
###############################################################################
###############################################################################
###############################################################################
class ToTensor_aux(object):

    def __call__(self, sample,  to_byte = False, is_gt_to_byte = False):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        
        rgb =  torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        depth =  torch.from_numpy(np.expand_dims(depth, 0)).float()
        
        gt0 = transform.rescale(gt,1/2, 
                                order=0, 
                                mode='reflect', 
                                preserve_range=True,
                                multichannel=False, 
                                anti_aliasing=False)
        gt1 = transform.rescale(gt,
                                1/4, 
                                order=0, 
                                mode='reflect', 
                                preserve_range=True,
                                multichannel=False, 
                                anti_aliasing=False)
        gt2 = transform.rescale(gt,
                                1/8, 
                                order=0, 
                                mode='reflect', 
                                preserve_range=True,
                                multichannel=False, 
                                anti_aliasing=False)
        gt =  torch.from_numpy(np.expand_dims(gt, 0)).float()
        gt0 = torch.from_numpy(np.expand_dims(gt0, 0)).float()
        gt1 = torch.from_numpy(np.expand_dims(gt1, 0)).float()
        gt2 = torch.from_numpy(np.expand_dims(gt2, 0)).float()
        return  {'rgb': rgb, 'depth': depth, 'gt': gt,'gt0':gt0,'gt1':gt1,'gt2':gt2, 'name':name}
class ToTensor_aux6(object):

    def __call__(self, sample,  to_byte = False, is_gt_to_byte = False):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        
        rgb =  torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        depth =  torch.from_numpy(np.expand_dims(depth, 0)).float()
        
        gt0 = transform.rescale(gt,1/2, 
                                order=0, 
                                mode='reflect', 
                                preserve_range=True,
                                multichannel=False, 
                                anti_aliasing=False)
        gt1 = transform.rescale(gt,1/4,
                                order=0, 
                                mode='reflect', 
                                preserve_range=True,
                                multichannel=False, 
                                anti_aliasing=False)
        gt2 = transform.rescale(gt,1/8,
                                order=0, 
                                mode='reflect', 
                                preserve_range=True,
                                multichannel=False, 
                                anti_aliasing=False)
        gt3 = transform.rescale(gt,1/16,
                                order=0, 
                                mode='reflect', 
                                preserve_range=True,
                                multichannel=False, 
                                anti_aliasing=False)
        gt =  torch.from_numpy(np.expand_dims(gt, 0)).float()
        gt0 = torch.from_numpy(np.expand_dims(gt0, 0)).float()
        gt1 = torch.from_numpy(np.expand_dims(gt1, 0)).float()
        gt2 = torch.from_numpy(np.expand_dims(gt2, 0)).float()
        gt3 = torch.from_numpy(np.expand_dims(gt3, 0)).float()
        return  {'rgb': rgb, 'depth': depth, 'gt': gt,'gt0':gt0,'gt1':gt1,'gt2':gt2,'gt3':gt3, 'name':name}
###############################################################################
###############################################################################
###############################################################################
class RandomHSV_aux(object):
    """
        Args:
            h_range (float tuple): random ratio of the hue channel,
                new_h range from h_range[0]*old_h to h_range[1]*old_h.
            s_range (float tuple): random ratio of the saturation channel,
                new_s range from s_range[0]*old_s to s_range[1]*old_s.
            v_range (int tuple): random bias of the value channel,
                new_v range from old_v-v_range to old_v+v_range.
        Notice:
            h range: 0-1
            s range: 0-1
            v range: 0-255
        """

    def __init__(self, h_range, s_range, v_range, p=0.25):
        assert isinstance(h_range, (list, tuple)) and \
               isinstance(s_range, (list, tuple)) and \
               isinstance(v_range, (list, tuple))
        self.p = p
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
    def __call__(self, sample):
        rgb, depth, gt,gt0, gt1, gt2, name = sample['rgb'], sample['depth'], sample['gt'], sample['gt0'], sample['gt1'], sample['gt2'], sample['name']
        rd = random.random()
        if  rd < self.p :
              rgb = self.randomhsv(rgb)
        return  {'rgb': rgb, 'depth': depth, 'gt': gt, 'gt0':gt0, 'gt1':gt1, 'gt2':gt2, 'name':name}
      
    def randomhsv(self, img):
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(-min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v + v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)
        return img_new  
###############################################################################
###############################################################################
###############################################################################
class zero_one_aux(object):
    def __call__(self, sample):
      rgb, depth, gt,gt0, gt1, gt2, name = sample['rgb'], sample['depth'], sample['gt'], sample['gt0'], sample['gt1'], sample['gt2'], sample['name']  
      rgb = rgb / 255.0
      depth = 1.0 - depth / 255.0
      gt = gt / 255.0
      
      gt0 = gt0 / 255.0
      gt1 = gt1 / 255.0
      gt2 = gt2 / 255.0
      return  {'rgb': rgb, 'depth': depth, 'gt': gt, 'gt0':gt0, 'gt1':gt1, 'gt2':gt2, 'name':name}

class zero_one_aux6(object):
    def __call__(self, sample):
      rgb, depth, gt,gt0, gt1, gt2, gt3, name = sample['rgb'], sample['depth'], sample['gt'], sample['gt0'], sample['gt1'], sample['gt2'], sample['gt3'], sample['name']  
      rgb = rgb / 255.0
      depth = depth / 255.0
      gt = gt / 255.0
      
      gt0 = gt0 / 255.0
      gt1 = gt1 / 255.0
      gt2 = gt2 / 255.0
      gt3 = gt3 / 255.0
      return  {'rgb': rgb, 'depth': depth, 'gt': gt,'gt0':gt0,'gt1':gt1,'gt2':gt2,'gt3':gt3, 'name':name}

    def depthp(self, depth):
            mean = torch.std(depth)
            std  = torch.mean(depth)
      
            depth = (depth-mean) / std
            return depth

###############################################################################
###############################################################################
###############################################################################
class per_image_standardization_aux(object):
      def __call__(self, sample):
            rgb, depth, gt,gt0, gt1, gt2, name = sample['rgb'], sample['depth'], sample['gt'], sample['gt0'], sample['gt1'], sample['gt2'], sample['name'] 
            
            rgb = self.image_standardization(rgb)
            depth = self.image_standardization(depth)
            
            return  {'rgb': rgb, 'depth': depth, 'gt': gt, 'gt0':gt0, 'gt1':gt1, 'gt2':gt2, 'name':name}
      
      def image_standardization(self, image_tensor):
            std = torch.std(image_tensor)
            mean = torch.mean(image_tensor)
            if std != 0:
                  image_tensor = (image_tensor - mean) / std
            return image_tensor
class per_image_standardization_aux6(object):
      def __call__(self, sample):
            rgb, depth, gt,gt0, gt1, gt2, gt3, name = sample['rgb'], sample['depth'], sample['gt'], sample['gt0'], sample['gt1'], sample['gt2'],sample['gt3'], sample['name'] 
            
            rgb = self.image_standardization(rgb)
            depth = self.image_standardization(depth)
            
            return  {'rgb': rgb, 'depth': depth, 'gt': gt, 'gt0':gt0, 'gt1':gt1, 'gt2':gt2,'gt3':gt3, 'name':name}
      
      def image_standardization(self, image_tensor):
            std = torch.std(image_tensor)
            mean = torch.mean(image_tensor)
            if std != 0:
                  image_tensor = (image_tensor - mean) / std
            return image_tensor
###############################################################################
###############################################################################
###############################################################################     
class normal(object):
    def __init__(self, rgb_mean=[0.36282069693095326, 0.40101408745784822,  0.4160890523421466],      
                       rgb_std = [ 0.2108164337383974, 0.20662449721294468, 0.21706249876367184],
                       depth_mean = [0.465029319074],
                       depth_std = [0.242993543074]):
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.depth_mean = depth_mean
        self.depth_std =depth_std
    def __call__(self, sample):
            rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']    
            rgb = transforms.Normalize(rgb, self.rgb_mean, self.rgb_std)
            depth = transforms.Normalize(depth, self.depth_mean, self.depth_std)
            return {'rgb': rgb, 'depth': depth, 'gt': gt, 'name': name}    
      
      
def show_rgb(img_tensor):
    out = img_tensor.detach().numpy()
    out = out.transpose((1, 2, 0))
    plt.imshow(out)
    plt.show()
def show(img):
    plt.imshow(img)
    plt.show()      
def test2():
      re = read_RGBD('../dataset/nju2000/train_path.npy')
      sample = re.__getitem__(2)
      
      hsv = RandomHSV((0.85, 1.25),(0.8, 1.2),(20, 30),1)
      sample = hsv.__call__(sample)
      gray = RandomGray()
      sample = gray.__call__(sample)
      
#      norm = per_image_standardization()
#      
#      sample = norm.__call__(sample)
      rgb = sample['rgb']
      show(rgb/255)
#      to = ToTensor()
#      sample = to.__call__(sample)
#      zero = zero_one()
#      sample = zero.__call__(sample)
#      rgb = sample['rgb']
#      show_rgb(rgb)
#      print(rgb.shape)
if __name__ == '__main__':
    test2()
