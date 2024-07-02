import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime

import torch


def get_config():
    parser = argparse.ArgumentParser()
    '''Base'''

    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='bert',
                        choices=['bert', 'roberta'])

    '''Optimization'''
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)

    '''Environment'''
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--backend', default=False, action='store_true')#action='store_true'：这个参数的行为是将--backend参数的值设置为True。这意味着如果用户在命令行中提供了--backend参数，无论后面是否跟随值，--backend参数的值都会被设置为True
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--timestamp', type=int, default='{:.0f}{:03}'.format(time.time(), random.randint(0, 999)))


    parser.add_argument('--train_file',type=str,default='train.json')
    parser.add_argument('--test_file',type=str,default='test.json')
    parser.add_argument('--pretrained_weights',type=str,default='../model')
    parser.add_argument('--save_file',type=str,default='bert.params')
    args = parser.parse_args()
    args.device = torch.device(args.device)

    return args