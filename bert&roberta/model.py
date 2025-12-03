"""
BERT/RoBERTa模型定义
"""
import torch
import torch.nn as nn
from transformers import BertModel
from typing import Optional
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StanceDetectionModel(nn.Module):
    """立场检测模型基类"""
    
    def __init__(
        self,
        num_classes: int,
        pretrained_weights: str,
        freeze_layers: int = 8,
        dropout: float = 0.1
    ):
        """
        初始化模型
        
        Args:
            num_classes: 分类类别数
            pretrained_weights: 预训练模型路径
            freeze_layers: 冻结的层数（从底层开始）
            dropout: Dropout比率
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.pretrained_weights = pretrained_weights
        
        # 加载预训练BERT模型
        logger.info(f"正在加载预训练模型: {pretrained_weights}")
        self.bert = BertModel.from_pretrained(pretrained_weights)
        
        # 冻结部分层
        self._freeze_layers(freeze_layers)
        
        # 分类头
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
        
        logger.info(f"模型初始化完成，类别数: {num_classes}, 冻结层数: {freeze_layers}")
    
    def _freeze_layers(self, freeze_layers: int):
        """
        冻结BERT的部分层
        
        Args:
            freeze_layers: 要冻结的层数
        """
        # 首先冻结所有参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 解冻pooler层
        for param in self.bert.pooler.parameters():
            param.requires_grad = True
        
        # 解冻顶部的若干层
        total_layers = 12
        trainable_layers = total_layers - freeze_layers
        
        if trainable_layers > 0:
            for i in range(total_layers - 1, total_layers - 1 - trainable_layers, -1):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = True
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.bert.parameters())
        logger.info(f"BERT可训练参数: {trainable_params:,} / {total_params:,} "
                   f"({100 * trainable_params / total_params:.2f}%)")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            token_type_ids: Token类型ID
            
        Returns:
            分类logits
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用pooler输出
        pooled_output = outputs.pooler_output
        
        # 分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # 应用softmax
        probs = torch.softmax(logits, dim=1)
        
        return probs


class BERTStanceModel(StanceDetectionModel):
    """BERT立场检测模型"""
    
    def __init__(
        self,
        num_classes: int,
        pretrained_weights: str,
        freeze_layers: int = 8
    ):
        """
        初始化BERT模型
        
        Args:
            num_classes: 分类类别数
            pretrained_weights: 预训练模型路径
            freeze_layers: 冻结的层数
        """
        super().__init__(num_classes, pretrained_weights, freeze_layers)
        logger.info("BERT立场检测模型初始化完成")


class RoBERTaStanceModel(StanceDetectionModel):
    """RoBERTa立场检测模型"""
    
    def __init__(
        self,
        num_classes: int,
        pretrained_weights: str,
        freeze_layers: int = 6
    ):
        """
        初始化RoBERTa模型
        
        Args:
            num_classes: 分类类别数
            pretrained_weights: 预训练模型路径
            freeze_layers: 冻结的层数（RoBERTa通常冻结更少的层）
        """
        super().__init__(num_classes, pretrained_weights, freeze_layers)
        logger.info("RoBERTa立场检测模型初始化完成")


def create_model(
    model_name: str,
    num_classes: int,
    pretrained_weights: str,
    device: Optional[torch.device] = None
) -> StanceDetectionModel:
    """
    创建模型
    
    Args:
        model_name: 模型名称 ('bert' 或 'roberta')
        num_classes: 分类类别数
        pretrained_weights: 预训练模型路径
        device: 运行设备
        
    Returns:
        模型实例
        
    Raises:
        ValueError: 当模型名称不支持时
    """
    if model_name.lower() == 'bert':
        model = BERTStanceModel(num_classes, pretrained_weights, freeze_layers=8)
    elif model_name.lower() == 'roberta':
        model = RoBERTaStanceModel(num_classes, pretrained_weights, freeze_layers=6)
    else:
        raise ValueError(f"不支持的模型类型: {model_name}")
    
    # 移动到指定设备
    if device is not None:
        model = model.to(device)
        logger.info(f"模型已移动到设备: {device}")
    
    return model
