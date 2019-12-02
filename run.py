# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 23:42:39 2018

@author: Dell
"""
import os
import time
import argparse
argement = argparse.ArgumentParser(description='运行参数')
argement.add_argument('--run-type', default=0, type=int, metavar='N',help='run-type default (0)')
run_args = argement.parse_args()

def get_setting(models, device_num, epochs, dataset,train_time):
     setting = []
     #[53,d,130,'nprl2',1]
     for m, t in zip(models, train_time):
           for epoch in epochs:
                 setting.append([m, device_num, epoch, dataset, t])
     return setting
if __name__ == '__main__':

     k = run_args.run_type
    
     infence_string = "python infence_for_nprl.py --model-type {0[0]} --device-number {0[1]} \
--last-ckpt /media/hpc/data/work/1/result/nprl2/au/model_tpye_{0[0]}/{0[4]}/ckpt/ckpt_epoch_{0[2]}.00.pth"
     train_string ="python train_aux14_nprl.py --model-type {0[0]}                 \
                                                --loss-type {0[1]}                  \
                                                --device-number {0[2]}              \
                                                --epochs {0[3]}                     \
                                                --lr {0[4]}                         \
                                                --lr-decay-rate {0[5]}              \
                                                --lr-epoch-per-decay {0[6]}  "
     last_string = "--last-ckpt /media/hpc/data/work/1/result/{0[9]}/au/model_tpye_{0[0]}/{0[7]}/ckpt/ckpt_epoch_{0[8]}.00.pth "

    
     train_string_last = train_string + last_string
     
     train_string2  = "python train_aux14_nju200-4.py --model-type {0[0]}                 \
                                                --loss-type {0[1]}                  \
                                                --device-number {0[2]}              \
                                                --epochs {0[3]}                     \
                                                --lr {0[4]}                         \
                                                --lr-decay-rate {0[5]}              \
                                                --lr-epoch-per-decay {0[6]}  "
     train_string2_last = train_string2 + last_string
     infence_string2 = "python infence_for_nju2000-2.py --model-type {0[0]} --device-number {0[1]} \
--last-ckpt /media/hpc/data/work/1/result/nju2000-2/au/model_tpye_{0[0]}/{0[4]}/ckpt/ckpt_epoch_{0[2]}.00.pth"


     infence_string3 = "python infence_for_stereo.py --model-type {0[0]} --device-number {0[1]} \
--last-ckpt /media/hpc/data/work/1/result/nju2000-4/au/model_tpye_{0[0]}/{0[4]}/ckpt/ckpt_epoch_{0[2]}.00.pth"

     infence_string4 = "python infence_for_nju2000-3.py --model-type {0[0]} --device-number {0[1]} \
--last-ckpt /media/hpc/data/work/1/result/nju2000-3/au/model_tpye_{0[0]}/{0[4]}/ckpt/ckpt_epoch_{0[2]}.00.pth"

     train_string3  = "python train_nju2000_3.py --model-type {0[0]}                 \
                                                --loss-type {0[1]}                  \
                                                --device-number {0[2]}              \
                                                --epochs {0[3]}                     \
                                                --lr {0[4]}                         \
                                                --lr-decay-rate {0[5]}              \
                                                --lr-epoch-per-decay {0[6]}  --batch_size 1"     


     train_nju2000_4  = "python train_aux14_nju200-4.py --model-type {0[0]}                 \
                                                --loss-type {0[1]}                  \
                                                --device-number {0[2]}              \
                                                --epochs {0[3]}                     \
                                                --lr {0[4]}                         \
                                                --lr-decay-rate {0[5]}              \
                                                --lr-epoch-per-decay {0[6]}  --batch-size 1" 
     infence_nju2000_4 = "python infence_for_nju2000-4.py --model-type {0[0]} --device-number {0[1]} \
--last-ckpt /media/hpc/data/work/1/result/nju2000-4/au/model_tpye_{0[0]}/{0[4]}/ckpt/ckpt_epoch_{0[2]}.00.pth"
     infence_sip1 = "python infence_for_sip.py --model-type {0[0]} --device-number {0[1]} \
--last-ckpt /media/hpc/data/work/1/result/nju2000-4/au/model_tpye_{0[0]}/{0[4]}/ckpt/ckpt_epoch_{0[2]}.00.pth"
     infence_sip2 = "python infence_for_sip.py --model-type {0[0]} --device-number {0[1]} \
--last-ckpt /media/hpc/data/work/1/result/nju2000-2/au/model_tpye_{0[0]}/{0[4]}/ckpt/ckpt_epoch_{0[2]}.00.pth"

     if run_args.run_type == 1 :
           d = 3
           models     = [1]
           train_time = [1]
           epochs = [200]#[188, 189,191, 192,193, 288,289,291,292,293]
           setts = get_setting(models, d, epochs, 'sip', train_time )
           for setting in setts:
                 os.system(infence_sip1.format(setting))        

     elif run_args.run_type == 2 :
           d = 1
           setts = [   
                       [1, 15, d, 300, 0.002, 0.8, 30, 1, 285, 'nju2000-4'],
                    ]
           for setting in setts:
                 os.system(train_nju2000_4.format(setting))      



             
