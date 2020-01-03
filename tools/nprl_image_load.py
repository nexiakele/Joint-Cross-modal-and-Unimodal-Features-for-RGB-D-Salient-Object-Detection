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
import random
import matplotlib.image  as mpimg
###############################################################################
###############################################################################
###############################################################################
class read_RGBD():
    """Face Landmarks dataset."""

    def __init__(self, file_path,  transform=None):
        self.file_path = file_path
        self.path = np.load(file_path)
        self.transform = transform
    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
          
        names = self.path[idx]
        rgb = io.imread(names[0])
        depth =np.load(names[1]) 
        gt = io.imread(names[2], as_gray=True) 
        name = names[3]
        if len(rgb.shape) == 2:
              rgb = color.gray2rgb(rgb)
        if len(gt.shape) == 3:
              gt = color.rgb2gray(gt)
        if gt.max() < 2:
              gt = gt * 255
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
    def __init__(self, w = 256, h = 256):
        self.w = w
        self.h = h
    def __call__(self, sample):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        # Bi-linear
        rgb = transform.resize(rgb, (self.h, self.w),order=1, mode='reflect', preserve_range=True)
        # Nearest-neighbor
        depth = transform.resize(depth, (self.h, self.w),order=0, mode='reflect', preserve_range=True)
        gt = transform.resize(gt, (self.h, self.w),order=0, mode='reflect', preserve_range=True)
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
class ToTensor_aux6(object):

    def __call__(self, sample,  to_byte = False, is_gt_to_byte = False):
        rgb, depth, gt, name = sample['rgb'], sample['depth'], sample['gt'], sample['name']
        
        rgb =  torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        depth =  torch.from_numpy(np.expand_dims(depth, 0)).float()
        gt0 = transform.rescale(gt,1/2, order=0, mode='reflect', preserve_range=True)
        gt1 = transform.rescale(gt,1/4, order=0, mode='reflect', preserve_range=True)
        gt2 = transform.rescale(gt,1/8, order=0, mode='reflect', preserve_range=True)
        gt3 = transform.rescale(gt,1/16, order=0, mode='reflect', preserve_range=True)
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
class zero_one_aux6(object):
    def __call__(self, sample):
      rgb, depth, gt,gt0, gt1, gt2, gt3, name = sample['rgb'], sample['depth'], sample['gt'], sample['gt0'], sample['gt1'], sample['gt2'], sample['gt3'], sample['name']  
      rgb = rgb / 255.0
      gt = gt / 255.0
      gt0 = gt0 / 255.0
      gt1 = gt1 / 255.0
      gt2 = gt2 / 255.0
      gt3 = gt3 / 255.0
      return  {'rgb': rgb, 'depth': depth, 'gt': gt,'gt0':gt0,'gt1':gt1,'gt2':gt2,'gt3':gt3, 'name':name}
###############################################################################
###############################################################################
###############################################################################
def show_rgb(img_tensor):
    out = img_tensor.detach().numpy()
    out = out.transpose((1, 2, 0))
    plt.imshow(out)
    plt.show()
def show(img):
    plt.imshow(img)
    plt.show()      
def test2():
      re = read_RGBD('../../../dataset/nprl/train_path.npy')
      for i in range(10):
            
            sample = re.__getitem__(i)
            rs = Resize(256, 256)
            sample = rs.__call__(sample)
            rgb = sample['rgb']
            
#      show(rgb/255)
#      hsv = RandomHSV((0.85, 1.25),(0.8, 1.2),(20, 30),1)
#      sample = hsv.__call__(sample)
#      gray = RandomGray()
#      sample = gray.__call__(sample)
      
##      norm = per_image_standardization()
##      
##      sample = norm.__call__(sample)
#      rgb = sample['rgb']
#      show(rgb/255)
##      to = ToTensor()
##      sample = to.__call__(sample)
##      zero = zero_one()
##      sample = zero.__call__(sample)
##      rgb = sample['rgb']
##      show_rgb(rgb)
##      print(rgb.shape)
if __name__ == '__main__':
    test2()
