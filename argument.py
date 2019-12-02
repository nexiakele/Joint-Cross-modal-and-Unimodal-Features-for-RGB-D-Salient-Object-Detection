# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 22:06:29 2018

@author: Dell
"""
import argparse

parser = argparse.ArgumentParser(description='训练参数')
###############################################################################
###########################模型相关#############################################
###############################################################################
parser.add_argument('--name', default='base' , help='model name')
parser.add_argument('--remarks', default='辅助' , help='model name')

parser.add_argument('--model-type', default=1, type=int, metavar='N',help='model type (default: 0)')

parser.add_argument('--loss-type', default=1, type=int, metavar='N',
                    help='loss type (default: 2) ')
###############################################################################
#############################file path#########################################
###############################################################################
parser.add_argument('--dataset-name', default='nprl2' , help='dataset name')
parser.add_argument('--is-per-img-norm', action='store_true', default=False,
                    help='是否对图像进行归一化，默认 否')
parser.add_argument('--train-file-path', default='/media/hpc/data/work/dataset/dataset-name/train_path.npy', 
                    type=str, metavar='PATH', help='train file path')

parser.add_argument('--val-file-path', default='/media/hpc/data/work/dataset/dataset-name/val_path.npy', 
                    type=str, metavar='PATH', help='val file path')

parser.add_argument('--test-file-path', default='/media/hpc/data/work/dataset/dataset-name/test_path.npy', 
                    type=str, metavar='PATH', help='test file path')
###############################################################################
#############################设备情况###########################################
###############################################################################
parser.add_argument('--is-cuda', action='store_false', default=True,
                    help='enables CUDA training')

parser.add_argument('--device-number', default=0, type=int, metavar='N',
                    help='GPU的设备号 (default: 2) ')

###############################################################################
#############################随机种子，使训练结果确定#############################
###############################################################################
parser.add_argument('--seed', default=1, type=int, metavar='N',
                    help='random seed (default: 1)')
###############################################################################
##############################训练参数1#########################################
###############################################################################
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--momentum', default=0.95, type=float, metavar='M',help='momentum')
###############################################################################
parser.add_argument('--lr-decay-rate', default=0.8, type=float,
                    help='decay rate of learning rate (default: 0.5)')

parser.add_argument('--lr-epoch-per-decay', default=25, type=int,
                    help='epoch of per decay of learning rate (default: 25)')

parser.add_argument('--lr-epoch-decay', default=0, type=int,
                    help='经过多少个epoch之后学习率才会发生变化 (default: 100)')
###############################################################################
##############################训练参数2#########################################
###############################################################################
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run (default: 1500)')

parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='batch size (default: 4)')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
###############################################################################
###############################备份相关#########################################
###############################################################################
parser.add_argument('--last-ckpt', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--checkpoint', action='store_false', default=True,
                    help='Using Pytorch checkpoint or not')

parser.add_argument('--ckpt-dir', default='./ckpt', metavar='DIR',
                    help='path to save checkpoints')
parser.add_argument('--start-ckpt-epoch', default=100, type=int, metavar='N',
                    help='ckpt epoch (default: 100) ')
parser.add_argument('--ckpt-epoch', default=2, type=int, metavar='N',
                    help='ckpt epoch (default: 2) ')

parser.add_argument('--summary-dir', default='./summay', metavar='DIR',
                    help='summary dir')

parser.add_argument('--summary-epoch', default=25, type=int, metavar='N',
                    help='ckpt epoch (default: 2) ')
###############################################################################
##############################验证相关##########################################
###############################################################################
parser.add_argument('--is_val', action='store_true', default=False,
                    help='是否使用验证集')

parser.add_argument('--val-freq', default=2, type=int, metavar='N',
                    help='验证集验证频率 (default: 2) ')

parser.add_argument('--milestones', nargs='+', type=int)
###############################梯度打印#########################################
parser.add_argument('--is-print-grad', action='store_true', default=False,
                    help='是否打印梯度')
parser.add_argument('--require-grad', action='store_false', default=True,
                    help='')
parser.add_argument('--grad-key-list', nargs='+', help='grad-key-list', required=False,
                    default=['Encoder.block0.base.rgb_block0.0.weight'])

###############################################################################
##############################重训练相关########################################
###############################################################################
parser.add_argument('--retrain-with-opt', action='store_true', default=False,
                    help='重训练是否使用优化器')

parser.add_argument('--is-find-lr-aux', default=0, type=int, metavar='N',
                    help='搜索合适的lr的时候决定使用哪个模型')
###############################################################################
parser.add_argument('--string', default=' ' , help='string')
parser.add_argument('--aux-loss-weight',default=[1,1,1,1,1,1,1,1,1,1],nargs='+', type=float)

###############################################################################
###############################################################################
def args_post_process(args):
      #处理数据库的路径
      args.train_file_path=args.train_file_path.replace("dataset-name",args.dataset_name)
      args.val_file_path=args.val_file_path.replace("dataset-name",args.dataset_name)
      args.test_file_path=args.test_file_path.replace("dataset-name",args.dataset_name)
      #训练属性汇总
      args.string = args.__str__().replace(',', '\r\n')
      return args
args = parser.parse_args()
args = args_post_process(args)
if __name__ == '__main__':
      from tools.Tools import make_dir 
      args = args_post_process(args)
#      print(args)
      args.model_type = 30
#      make_dir(args)
      print(make_dir(args))
      
