# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:12:07 2018

@author: Dell
"""

import numpy as np
from torch.optim.lr_scheduler import _LRScheduler 

class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        max_lr :  最大学习率
        gamma (float): 指数衰减率，当model=2时生效.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.

    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups

        >>> scheduler = CyclicLR(optimizer,step_size=200, max_lr=0.5,epochs = 0,model = 2)
        >>> for step in range(20000):
        >>>     scheduler.step(step)
        >>>     scheduler.set_epochs(step // 200)
        >>>     
    """
    def __init__(self, optimizer, step_size, max_lr, gamma = 0.9, epochs = 0, model = 0):
        self.step_size = step_size
        self.max_lr = max_lr
        self.gamma =gamma
        self.model = model
        self.epochs = epochs
        super(CyclicLR, self).__init__(optimizer, last_epoch = -1)
    def get_lr(self):
        return [ self.get_lr_tri(base_lr) for base_lr in self.base_lrs]
        
    def get_lr_tri(self, base_lr):
          cycle = np.floor(1+self.last_epoch/(2*self.step_size))
          x = np.abs(self.last_epoch/self.step_size - 2*cycle + 1)
          if self.model == 0: # 'triangular'
                lr = base_lr + (self.max_lr-base_lr)*np.maximum(0, (1-x))
          elif self.model == 1: # 'triangular2'
                lr = base_lr + (self.max_lr-base_lr)*np.maximum(0, (1-x))/float(2**(cycle-1))
          elif self.model == 2: # 'triangular exp'
                lr= base_lr + \
                    (self.max_lr-base_lr)*np.maximum(0, (1-x))*self.gamma**(self.epochs)
          elif self.model == 3: # 'triangular exp'
                lr= (base_lr+(self.max_lr-base_lr)*np.maximum(0,(1-x)))*self.gamma**(self.epochs)
          return lr
    def set_epochs(self, epochs):
          self.epochs = epochs

#import matplotlib.pyplot as plt
#import torch.optim as optim
#if __name__ == '__main__':
#      net = models.resnet18()
#      optimizer = optim.SGD(net.parameters(), lr=10, momentum=0.9)
#      scheduler = CyclicLR(optimizer,step_size=200, max_lr=20,epochs = 0,model = 3)
#      lrs = []
#      global_step = 0
#      for epoch in range(40):
#            scheduler.set_epochs(max(0,(epoch-20) // 2))
#            for step in range(1000):
#                  global_step+=1
#                  scheduler.step(global_step)
#                  
#                  lr = optimizer.param_groups[0]['lr']
#                  lrs.append(lr)
#      plt.plot(lrs)
#      plt.show()