"""
T5模型配置文件
"""
import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional


class T5Config:
    """T5模型配置类"""
    
    def __init__(
        self,
        # 模型参数
        pretrained_model_name_or_path: str = './mt5model',
        classes_map_path: str = 'classes_map.json',
        prefix_text: str = "判断评论立场：",
        
        # 训练参数
        num_train_epochs: int = 12,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        lr_warmup_steps: int = 0,
        
        # 优化器配置
        use_adafactor: bool = True,
        use_adafactor_schedule: bool = True,
        
        # 数据参数
        data_dir: str = None,
        use_weighted_sampler: bool = False,
        
        # 环境参数
        device: Optional[str] = None,
        num_workers: int = 0,
        save_weights_path: str = 'weights',
        
        # 其他
        seed: int = 42
    ):
        """
        初始化T5配置
        
        Args:
            pretrained_model_name_or_path: 预训练模型路径
            classes_map_path: 类别映射文件路径
            prefix_text: 输入文本前缀
            num_train_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            weight_decay: 权重衰减
            lr_warmup_steps: 学习率预热步数
            use_adafactor: 是否使用Adafactor优化器
            use_adafactor_schedule: 是否使用Adafactor调度器
            data_dir: 数据目录
            use_weighted_sampler: 是否使用加权采样
            device: 运行设备
            num_workers: 数据加载线程数
            save_weights_path: 模型权重保存目录
            seed: 随机种子
        """
        # 模型参数
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        
        # 类别映射文件路径
        if os.path.isabs(classes_map_path):
            self.classes_map_path = classes_map_path
        else:
            self.classes_map_path = os.path.join(sys.path[0], classes_map_path)
        
        self.prefix_text = prefix_text
        
        # 训练参数
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_warmup_steps = lr_warmup_steps
        
        # 优化器配置
        self.use_adafactor = use_adafactor
        self.use_adafactor_schedule = use_adafactor_schedule
        
        # 数据参数
        if data_dir is None:
            self.data_dir = os.path.join(sys.path[0], 'dataset')
        else:
            self.data_dir = data_dir
        
        self.use_weighted_sampler = use_weighted_sampler
        
        # 环境参数
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # 自动设置num_workers
        if num_workers == 0:
            self.num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
            self.num_workers = 0  # Windows系统建议设为0
        else:
            self.num_workers = num_workers
        
        # 保存路径
        if os.path.isabs(save_weights_path):
            self.save_weights_path = save_weights_path
        else:
            self.save_weights_path = os.path.join(sys.path[0], save_weights_path)
        
        self.seed = seed
        
        # 创建保存目录
        os.makedirs(self.save_weights_path, exist_ok=True)
    
    def get_data_files(self):
        """获取数据文件路径"""
        return {
            'train': os.path.join(self.data_dir, 'train.json'),
            'validation': os.path.join(self.data_dir, 'validation.json'),
            'test': os.path.join(self.data_dir, 'test.json')
        }
    
    def __repr__(self):
        """返回配置的字符串表示"""
        config_dict = {
            '模型配置': {
                '预训练模型': self.pretrained_model_name_or_path,
                '类别映射文件': self.classes_map_path,
                '输入前缀': self.prefix_text,
            },
            '训练配置': {
                '训练轮数': self.num_train_epochs,
                '批次大小': self.batch_size,
                '学习率': self.learning_rate,
                '权重衰减': self.weight_decay,
                '预热步数': self.lr_warmup_steps,
                '使用Adafactor': self.use_adafactor,
                '使用Adafactor调度': self.use_adafactor_schedule,
            },
            '数据配置': {
                '数据目录': self.data_dir,
                '加权采样': self.use_weighted_sampler,
            },
            '环境配置': {
                '设备': str(self.device),
                '工作线程': self.num_workers,
                '保存路径': self.save_weights_path,
                '随机种子': self.seed,
            }
        }
        
        lines = ["=" * 60, "T5模型配置信息", "=" * 60]
        for category, params in config_dict.items():
            lines.append(f"\n{category}:")
            for key, value in params.items():
                lines.append(f"  {key}: {value}")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def get_config_from_args() -> T5Config:
    """
    从命令行参数获取配置
    
    Returns:
        T5Config对象
    """
    parser = argparse.ArgumentParser(description='T5立场检测模型训练/测试')
    
    # 模型参数
    parser.add_argument('--pretrained_model_name_or_path', type=str, 
                       default='./mt5model',
                       help='预训练模型路径')
    parser.add_argument('--classes_map_path', type=str,
                       default='classes_map.json',
                       help='类别映射文件路径')
    parser.add_argument('--prefix_text', type=str,
                       default='判断评论立场：',
                       help='输入文本前缀')
    
    # 训练参数
    parser.add_argument('--num_train_epochs', type=int, default=12,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='权重衰减')
    parser.add_argument('--lr_warmup_steps', type=int, default=0,
                       help='学习率预热步数')
    
    # 优化器配置
    parser.add_argument('--use_adafactor', action='store_true', default=True,
                       help='是否使用Adafactor优化器')
    parser.add_argument('--use_adafactor_schedule', action='store_true', default=True,
                       help='是否使用Adafactor调度器')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据目录')
    parser.add_argument('--use_weighted_sampler', action='store_true',
                       help='是否使用加权随机采样')
    
    # 环境参数
    parser.add_argument('--device', type=str, default=None,
                       help='运行设备')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='数据加载线程数')
    parser.add_argument('--save_weights_path', type=str, default='weights',
                       help='模型权重保存目录')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 测试专用参数
    parser.add_argument('--weights_name', type=str, default=None,
                       help='要加载的权重文件名（仅测试时使用）')
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = T5Config(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        classes_map_path=args.classes_map_path,
        prefix_text=args.prefix_text,
        num_train_epochs=args.num_train_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_warmup_steps=args.lr_warmup_steps,
        use_adafactor=args.use_adafactor,
        use_adafactor_schedule=args.use_adafactor_schedule,
        data_dir=args.data_dir,
        use_weighted_sampler=args.use_weighted_sampler,
        device=args.device,
        num_workers=args.num_workers,
        save_weights_path=args.save_weights_path,
        seed=args.seed
    )
    
    # 添加weights_name属性（仅用于测试）
    if hasattr(args, 'weights_name') and args.weights_name:
        config.weights_name = args.weights_name
    
    return config

