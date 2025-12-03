"""
T5模型训练模块
"""
import sys
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdafactorSchedule, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5Trainer:
    """T5训练器"""
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        label_id_list: List[int],
        device: torch.device,
        num_epochs: int = 12,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        use_adafactor: bool = True,
        use_adafactor_schedule: bool = True
    ):
        """
        初始化训练器
        
        Args:
            model: T5模型
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            label_id_list: 标签ID列表
            device: 运行设备
            num_epochs: 训练轮数
            learning_rate: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            use_adafactor: 是否使用Adafactor优化器
            use_adafactor_schedule: 是否使用Adafactor调度器
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.label_id_list = label_id_list
        self.device = device
        self.num_epochs = num_epochs
        
        # 配置优化器和调度器
        if use_adafactor and use_adafactor_schedule:
            # Adafactor with adaptive learning rate
            self.optimizer = Adafactor(
                model.parameters(),
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=None
            )
            self.scheduler = AdafactorSchedule(self.optimizer)
            logger.info("使用Adafactor优化器（自适应学习率）")
            
        elif use_adafactor and not use_adafactor_schedule:
            # Adafactor with fixed learning rate
            self.optimizer = Adafactor(
                model.parameters(),
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=learning_rate
            )
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=len(train_loader) * num_epochs
            )
            logger.info("使用Adafactor优化器（固定学习率）")
            
        else:
            # AdamW optimizer
            self.optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            self.scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=len(train_loader) * num_epochs
            )
            logger.info("使用AdamW优化器")
        
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
        
        predicted_labels = []
        ground_truth_labels = []
        total_loss = 0.0
        
        # 使用tqdm显示进度
        pbar = tqdm(self.train_loader, desc=f"[训练 Epoch {epoch}]", file=sys.stdout)
        
        for step, data in enumerate(pbar):
            # 移动到设备
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            labels = data['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            
            # 获取预测标签（取第一个token的预测）
            pred_labels = torch.max(logits, dim=-1).indices[:, 0]
            
            # 收集标签
            ground_truth_labels.extend(labels[:, 0].cpu().tolist())
            predicted_labels.extend(pred_labels.cpu().tolist())
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 更新学习率
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算当前准确率
            accuracy = accuracy_score(ground_truth_labels, predicted_labels)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{total_loss / (step + 1):.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # 计算平均指标
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(ground_truth_labels, predicted_labels)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            包含各项指标的字典
        """
        self.model.eval()
        
        predicted_labels = []
        ground_truth_labels = []
        
        # 使用tqdm显示进度
        pbar = tqdm(self.valid_loader, desc=f"[验证 Epoch {epoch}]", file=sys.stdout)
        
        for step, data in enumerate(pbar):
            # 移动到设备
            input_ids = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            labels = data['labels'].to(self.device)
            
            # 生成预测
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=2
            )
            
            pred_labels = outputs[:, 1]  # 取第二个token（第一个是开始符）
            
            # 检查是否有标签外的预测
            for pred in pred_labels:
                if pred.item() not in self.label_id_list:
                    logger.warning(f"预测标签不在标签列表中: {pred.item()}")
            
            # 收集标签
            ground_truth_labels.extend(labels[:, 0].cpu().tolist())
            predicted_labels.extend(pred_labels.cpu().tolist())
            
            # 计算当前指标
            accuracy = accuracy_score(ground_truth_labels, predicted_labels)
            macro_f1 = f1_score(ground_truth_labels, predicted_labels, average='macro')
            
            # 更新进度条
            pbar.set_postfix({
                'acc': f'{accuracy:.4f}',
                'macro_f1': f'{macro_f1:.4f}'
            })
        
        # 计算最终指标
        accuracy = accuracy_score(ground_truth_labels, predicted_labels)
        macro_f1 = f1_score(ground_truth_labels, predicted_labels, average='macro')
        micro_f1 = f1_score(ground_truth_labels, predicted_labels, average='micro')
        weighted_f1 = f1_score(ground_truth_labels, predicted_labels, average='weighted')
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1
        }
    
    def train(self, save_path: str = None) -> Dict[str, float]:
        """
        完整训练流程
        
        Args:
            save_path: 模型保存路径（不含文件名）
            
        Returns:
            最佳验证结果
        """
        best_macro_f1 = 0.0
        best_epoch = 0
        
        logger.info(f"开始训练，共{self.num_epochs}个epoch")
        
        for epoch in range(1, self.num_epochs + 1):
            # 训练
            train_result = self.train_epoch(epoch)
            
            # 验证
            valid_result = self.validate(epoch)
            
            # 打印结果
            logger.info("=" * 80)
            logger.info(f"Epoch {epoch}/{self.num_epochs}")
            logger.info(f"  学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            logger.info(f"  训练损失: {train_result['loss']:.4f}")
            logger.info(f"  训练准确率: {train_result['accuracy']:.4f}")
            logger.info(f"  验证准确率: {valid_result['accuracy']:.4f}")
            logger.info(f"  验证Macro F1: {valid_result['macro_f1']:.4f}")
            logger.info(f"  验证Micro F1: {valid_result['micro_f1']:.4f}")
            logger.info(f"  验证Weighted F1: {valid_result['weighted_f1']:.4f}")
            
            # 保存最佳模型
            if valid_result['macro_f1'] > best_macro_f1:
                best_macro_f1 = valid_result['macro_f1']
                best_epoch = epoch
                
                if save_path:
                    import os
                    from datetime import datetime
                    
                    model_name = os.path.basename(self.model.config.name_or_path)
                    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
                    save_file = os.path.join(
                        save_path,
                        f'{model_name}-{current_time}-epoch{epoch}.pth'
                    )
                    
                    torch.save(self.model.state_dict(), save_file)
                    logger.info(f"  保存最佳模型: {save_file}")
        
        logger.info("=" * 80)
        logger.info(f"训练完成! 最佳Epoch: {best_epoch}, 最佳Macro F1: {best_macro_f1:.4f}")
        
        return {
            'best_epoch': best_epoch,
            'best_macro_f1': best_macro_f1
        }

