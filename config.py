"""
BERT/RoBERTa模型配置文件
"""
import argparse
import torch
from pathlib import Path
from typing import Optional


class ModelConfig:
    """模型配置类"""
    
    def __init__(
        self,
        # 模型参数
        model_name: str = 'bert',
        num_classes: int = 3,
        pretrained_weights: str = '../model',
        
        # 训练参数
        train_batch_size: int = 16,
        test_batch_size: int = 32,
        num_epochs: int = 50,
        learning_rate: float = 4e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 25,
        total_training_steps: int = 40,
        
        # 数据参数
        max_length: int = 500,
        train_file: str = 'train.json',
        test_file: str = 'test.json',
        
        # 环境参数
        device: Optional[str] = None,
        num_workers: int = 0,
        save_file: str = 'bert.params',
        seed: int = 42
    ):
        """
        初始化配置
        
        Args:
            model_name: 模型名称，'bert'或'roberta'
            num_classes: 分类类别数
            pretrained_weights: 预训练模型路径
            train_batch_size: 训练批次大小
            test_batch_size: 测试批次大小
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            total_training_steps: 总训练步数
            max_length: 最大序列长度
            train_file: 训练数据文件
            test_file: 测试数据文件
            device: 运行设备
            num_workers: 数据加载线程数
            save_file: 模型保存文件
            seed: 随机种子
        """
        # 模型参数
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights
        
        # 训练参数
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_training_steps = total_training_steps
        
        # 数据参数
        self.max_length = max_length
        self.train_file = train_file
        self.test_file = test_file
        
        # 环境参数
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.num_workers = num_workers
        self.save_file = save_file
        self.seed = seed
        
        # 验证配置
        self._validate()
    
    def _validate(self):
        """验证配置参数"""
        if self.model_name not in ['bert', 'roberta']:
            raise ValueError(f"不支持的模型类型: {self.model_name}，仅支持'bert'或'roberta'")
        
        if self.num_classes <= 0:
            raise ValueError(f"类别数必须大于0，当前值: {self.num_classes}")
        
        if not Path(self.pretrained_weights).exists():
            print(f"警告: 预训练模型路径不存在: {self.pretrained_weights}")
    
    def __repr__(self):
        """返回配置的字符串表示"""
        config_dict = {
            '模型配置': {
                '模型名称': self.model_name,
                '类别数': self.num_classes,
                '预训练权重': self.pretrained_weights,
            },
            '训练配置': {
                '训练批次大小': self.train_batch_size,
                '测试批次大小': self.test_batch_size,
                '训练轮数': self.num_epochs,
                '学习率': self.learning_rate,
                '权重衰减': self.weight_decay,
            },
            '数据配置': {
                '最大长度': self.max_length,
                '训练文件': self.train_file,
                '测试文件': self.test_file,
            },
            '环境配置': {
                '设备': str(self.device),
                '工作线程': self.num_workers,
                '保存文件': self.save_file,
                '随机种子': self.seed,
            }
        }
        
        lines = ["=" * 50, "模型配置信息", "=" * 50]
        for category, params in config_dict.items():
            lines.append(f"\n{category}:")
            for key, value in params.items():
                lines.append(f"  {key}: {value}")
        lines.append("=" * 50)
        
        return "\n".join(lines)


def get_config_from_args() -> ModelConfig:
    """
    从命令行参数获取配置
    
    Returns:
        ModelConfig对象
    """
    parser = argparse.ArgumentParser(description='BERT/RoBERTa立场检测模型训练/测试')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert',
                        choices=['bert', 'roberta'],
                        help='模型名称')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='分类类别数')
    parser.add_argument('--pretrained_weights', type=str, default='../model',
                        help='预训练模型路径')
    
    # 训练参数
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='训练批次大小')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='测试批次大小')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=4e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='权重衰减')
    
    # 数据参数
    parser.add_argument('--max_length', type=int, default=500,
                        help='最大序列长度')
    parser.add_argument('--train_file', type=str, default='train.json',
                        help='训练数据文件')
    parser.add_argument('--test_file', type=str, default='test.json',
                        help='测试数据文件')
    
    # 环境参数
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载线程数')
    parser.add_argument('--save_file', type=str, default='bert.params',
                        help='模型保存文件')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 创建配置对象
    config = ModelConfig(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained_weights=args.pretrained_weights,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        train_file=args.train_file,
        test_file=args.test_file,
        device=args.device,
        num_workers=args.num_workers,
        save_file=args.save_file,
        seed=args.seed
    )
    
    return config

