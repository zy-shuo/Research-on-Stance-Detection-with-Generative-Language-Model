"""
数据加载模块
处理立场检测数据集的加载和预处理
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict, Tuple, Any, Optional
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载数据
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        数据列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"成功加载数据: {file_path}, 共{len(data)}条")
        return data
    except Exception as e:
        logger.error(f"加载数据失败: {file_path}, 错误: {str(e)}")
        raise


class StanceDetectionDataset(Dataset):
    """立场检测数据集类"""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: BertTokenizer,
        model_name: str = 'bert',
        max_length: int = 500,
        use_background: bool = False,
        use_labels: bool = False
    ):
        """
        初始化数据集
        
        Args:
            data: 数据列表
            tokenizer: 分词器
            model_name: 模型名称 ('bert' 或 'roberta')
            max_length: 最大序列长度
            use_background: 是否使用背景信息
            use_labels: 是否使用标签信息
        """
        self.data = data
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_length = max_length
        self.use_background = use_background
        self.use_labels = use_labels
        
        logger.info(f"数据集初始化完成，共{len(data)}条数据")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, index: int) -> Tuple[str, int]:
        """
        获取单条数据
        
        Args:
            index: 数据索引
            
        Returns:
            (文本, 标签)元组
        """
        item = self.data[index]
        label = item["label"]
        
        # 构建输入文本
        text = self._build_text(item)
        
        return text, label
    
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
            sep = '[SEP]' if self.model_name == 'bert' else '</s>'
            text += sep + '背景:' + item["简约背景"]
        
        # 可选：添加标签信息
        if self.use_labels and "全部立场标签" in item:
            sep = '[SEP]' if self.model_name == 'bert' else '</s>'
            text += sep + item["全部立场标签"]
        
        return text
    
    def collate_fn(self, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, ...]:
        """
        批处理函数
        
        Args:
            batch: 批次数据
            
        Returns:
            (input_ids, attention_mask, token_type_ids, labels)元组
        """
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        
        # 使用tokenizer进行编码
        encoded = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_length=True
        )
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded['token_type_ids']
        labels = torch.LongTensor(labels)
        
        return input_ids, attention_mask, token_type_ids, labels


def create_data_loader(
    data_path: str,
    tokenizer: BertTokenizer,
    model_name: str,
    batch_size: int,
    max_length: int = 500,
    shuffle: bool = True,
    num_workers: int = 0,
    use_background: bool = False,
    use_labels: bool = False
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_path: 数据文件路径
        tokenizer: 分词器
        model_name: 模型名称
        batch_size: 批次大小
        max_length: 最大序列长度
        shuffle: 是否打乱数据
        num_workers: 工作线程数
        use_background: 是否使用背景信息
        use_labels: 是否使用标签信息
        
    Returns:
        DataLoader对象
    """
    # 加载数据
    data = load_json_data(data_path)
    
    # 创建数据集
    dataset = StanceDetectionDataset(
        data=data,
        tokenizer=tokenizer,
        model_name=model_name,
        max_length=max_length,
        use_background=use_background,
        use_labels=use_labels
    )
    
    # 创建数据加载器
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle  # 训练时丢弃最后不完整的batch
    )
    
    logger.info(f"数据加载器创建完成，批次大小: {batch_size}, 共{len(data_loader)}个批次")
    
    return data_loader

