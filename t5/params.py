import os
import sys
import torch


classes_map_dir = os.path.join(sys.path[0], "classes_map.json")
prefix_text = "判断评论立场："

pretrained_model_name_or_path = './mt5model'

num_train_epochs = 12
batch_size = 4
learning_rate = 2e-4
lr_warmup_steps = 0
weight_decay = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
num_workers=0
data_dir = os.path.join(sys.path[0], 'dataset')  # 数据集目录
save_weights_path = os.path.join(sys.path[0], 'weights')  # 模型权重保存目录

