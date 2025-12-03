"""
模型训练模块
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_scheduler
from typing import Dict, Optional
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器类"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 4e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 25,
        total_steps: int = 40,
        num_epochs: int = 50
    ):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_loader: 训练数据加载器
            device: 运行设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            total_steps: 总步数
            num_epochs: 训练轮数
        """
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.num_epochs = num_epochs
        
        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 学习率调度器
        self.scheduler = get_scheduler(
            name='linear',
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            optimizer=self.optimizer
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        logger.info("训练器初始化完成")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            包含损失和准确率的字典
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm显示进度
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for batch_idx, (input_ids, attention_mask, token_type_ids, labels) in enumerate(pbar):
            # 移动到设备
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算损失
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 更新学习率
        self.scheduler.step()
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        logger.info(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def train(self, save_path: Optional[str] = None) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            save_path: 模型保存路径
            
        Returns:
            训练历史记录
        """
        history = {
            'loss': [],
            'accuracy': [],
            'learning_rate': []
        }
        
        logger.info(f"开始训练，共{self.num_epochs}个epoch")
        
        for epoch in range(1, self.num_epochs + 1):
            metrics = self.train_epoch(epoch)
            
            # 记录历史
            history['loss'].append(metrics['loss'])
            history['accuracy'].append(metrics['accuracy'])
            history['learning_rate'].append(metrics['learning_rate'])
        
        # 保存模型
        if save_path:
            torch.save(self.model.state_dict(), save_path)
            logger.info(f"模型已保存到: {save_path}")
        
        logger.info("训练完成")
        
        return history


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float = 4e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 25,
    total_steps: int = 40,
    save_path: Optional[str] = None
) -> Dict[str, list]:
    """
    训练模型的便捷函数
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        device: 运行设备
        num_epochs: 训练轮数
        learning_rate: 学习率
        weight_decay: 权重衰减
        warmup_steps: 预热步数
        total_steps: 总步数
        save_path: 模型保存路径
        
    Returns:
        训练历史记录
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        num_epochs=num_epochs
    )
    
    return trainer.train(save_path)

