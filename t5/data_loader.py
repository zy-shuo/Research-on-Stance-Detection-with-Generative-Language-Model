"""
T5数据加载模块
"""
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import T5Tokenizer
from typing import List, Dict, Any, Optional
import logging

from data_utils import load_json, validate_data_format, get_label_distribution


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5StanceDataset(Dataset):
    """T5立场检测数据集"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: T5Tokenizer,
        classes_map: Dict[str, int],
        prefix_text: str = "判断评论立场：",
        use_background: bool = False,
        use_labels: bool = False
    ):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            tokenizer: T5分词器
            classes_map: 类别映射字典 {类别名: 标签ID}
            prefix_text: 输入文本前缀
            use_background: 是否使用背景信息
            use_labels: 是否使用标签信息
        """
        self.data = data
        self.tokenizer = tokenizer
        self.prefix_text = prefix_text
        self.use_background = use_background
        self.use_labels = use_labels
        
        # 创建标签到类别名的映射
        self.labels_map = {value: key for key, value in classes_map.items()}
        
        # 验证数据格式
        if not validate_data_format(data):
            raise ValueError("数据格式验证失败")
        
        # 统计标签分布
        get_label_distribution(data)
        
        logger.info(f"数据集初始化完成，共{len(data)}条数据")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        获取单条数据
        
        Args:
            index: 数据索引
            
        Returns:
            包含input_ids, attention_mask, labels的字典
        """
        item = self.data[index]
        label = item["label"]
        
        # 构建输入文本
        text = self._build_text(item)
        
        # 编码输入文本
        text_encoded = self.tokenizer.encode_plus(
            text=self.prefix_text + text,
            add_special_tokens=True,  # T5没有[CLS]和[SEP]，有结束符</s>
            return_attention_mask=True
        )
        
        # 编码标签（转换为类别名）
        label_text = self.labels_map[label]
        labels_encoded = self.tokenizer.encode(
            text=label_text,
            add_special_tokens=True  # 在末尾添加 </s>
        )
        
        return {
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "labels": labels_encoded
        }
    
    def _build_text(self, item: Dict[str, Any]) -> str:
        """
        构建输入文本
        
        Args:
            item: 数据项
            
        Returns:
            构建好的文本
        """
        # 基础文本：评论 + 目标
        text = item["text"] + ' ' + '目标:' + item["target"]
        
        # 可选：添加背景信息
        if self.use_background and "简约背景" in item:
            text += ' ' + '背景:' + item["简约背景"]
        
        # 可选：添加标签信息
        if self.use_labels and "全部立场标签" in item:
            text += ' ' + item["全部立场标签"]
        
        return text
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        批处理函数
        
        Args:
            batch: 批次数据
            
        Returns:
            包含input_ids, attention_mask, labels的字典
        """
        # 提取各个字段
        input_ids_list = [torch.tensor(instance['input_ids']) for instance in batch]
        attention_mask_list = [torch.tensor(instance['attention_mask']) for instance in batch]
        labels_list = [torch.tensor(instance['labels']) for instance in batch]
        
        # 填充到相同长度
        input_ids_pad = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        attention_mask_pad = pad_sequence(
            attention_mask_list,
            batch_first=True,
            padding_value=0
        )
        
        labels_pad = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        
        return {
            "input_ids": input_ids_pad,
            "attention_mask": attention_mask_pad,
            "labels": labels_pad
        }


def create_data_loader(
    data_path: str,
    tokenizer: T5Tokenizer,
    classes_map: Dict[str, int],
    prefix_text: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    use_weighted_sampler: bool = False,
    use_background: bool = False,
    use_labels: bool = False
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        tokenizer: T5分词器
        classes_map: 类别映射字典
        prefix_text: 输入文本前缀
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作线程数
        use_weighted_sampler: 是否使用加权采样
        use_background: 是否使用背景信息
        use_labels: 是否使用标签信息
        
    Returns:
        DataLoader对象
    """
    # 加载数据
    data = load_json(data_path)
    
    # 创建数据集
    dataset = T5StanceDataset(
        data=data,
        tokenizer=tokenizer,
        classes_map=classes_map,
        prefix_text=prefix_text,
        use_background=use_background,
        use_labels=use_labels
    )
    
    # 创建采样器
    sampler = None
    if use_weighted_sampler and shuffle:
        logger.info("创建加权随机采样器...")
        
        # 统计每个标签的数量
        label_counts = {}
        for item in data:
            label = item['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 计算每个样本的权重
        weights = []
        for item in data:
            label = item['label']
            weight = 1.0 / label_counts[label]
            weights.append(weight)
        
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(data),
            replacement=False
        )
        
        shuffle = False  # 使用sampler时不能shuffle
        logger.info("加权随机采样器创建完成")
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        drop_last=shuffle and sampler is None  # 训练时丢弃最后不完整的batch
    )
    
    logger.info(f"数据加载器创建完成，批次大小: {batch_size}, 共{len(data_loader)}个批次")
    
    return data_loader

