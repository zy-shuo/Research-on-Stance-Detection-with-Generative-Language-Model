"""
模型评估模块
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """评估器类"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        event_names: List[str] = None
    ):
        """
        初始化评估器
        
        Args:
            model: 模型
            test_loader: 测试数据加载器
            device: 运行设备
            event_names: 事件名称列表（用于分事件统计）
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.event_names = event_names or [
            "恶意殴打他人者的妻女被网暴",
            "女子不让6岁男童上女厕所遭痛骂",
            "警方通告胡鑫宇为自杀",
            "满江红起诉大V",
            "泼水节女生选择原谅对方"
        ]
        
        logger.info("评估器初始化完成")
    
    def predict(self) -> Tuple[List[int], List[int]]:
        """
        对测试集进行预测
        
        Returns:
            (预测标签列表, 真实标签列表)
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        
        logger.info("开始预测...")
        
        with torch.no_grad():
            for input_ids, attention_mask, token_type_ids, labels in tqdm(self.test_loader):
                # 移动到设备
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                
                # 预测
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                predictions = outputs.argmax(dim=1).cpu().tolist()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.tolist())
        
        logger.info(f"预测完成，共{len(all_predictions)}条数据")
        
        return all_predictions, all_labels
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型性能
        
        Returns:
            包含各项指标的字典
        """
        predictions, labels = self.predict()
        
        # 计算整体指标
        accuracy = accuracy_score(labels, predictions)
        macro_f1 = f1_score(labels, predictions, labels=[0, 1, 2], average='macro')
        micro_f1 = f1_score(labels, predictions, labels=[0, 1, 2], average='micro')
        
        # 计算每个类别的F1分数
        f1_per_class = f1_score(labels, predictions, labels=[0, 1, 2], average=None)
        
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'f1_favor': f1_per_class[0],
            'f1_against': f1_per_class[1],
            'f1_neutral': f1_per_class[2]
        }
        
        # 打印整体结果
        logger.info("=" * 60)
        logger.info("整体评估结果:")
        logger.info(f"  准确率 (Accuracy): {accuracy:.4f}")
        logger.info(f"  宏平均F1 (Macro F1): {macro_f1:.4f}")
        logger.info(f"  微平均F1 (Micro F1): {micro_f1:.4f}")
        logger.info(f"  支持类F1: {f1_per_class[0]:.4f}")
        logger.info(f"  反对类F1: {f1_per_class[1]:.4f}")
        logger.info(f"  中立类F1: {f1_per_class[2]:.4f}")
        
        # 计算分事件指标
        if len(predictions) == 500:  # 假设每个事件100条数据
            logger.info("=" * 60)
            logger.info("分事件评估结果:")
            
            for i, event_name in enumerate(self.event_names):
                start_idx = i * 100
                end_idx = (i + 1) * 100
                
                event_predictions = predictions[start_idx:end_idx]
                event_labels = labels[start_idx:end_idx]
                
                event_macro_f1 = f1_score(
                    event_labels,
                    event_predictions,
                    labels=[0, 1, 2],
                    average='macro'
                )
                event_micro_f1 = f1_score(
                    event_labels,
                    event_predictions,
                    labels=[0, 1, 2],
                    average='micro'
                )
                
                logger.info(f"  {event_name}:")
                logger.info(f"    Macro F1: {event_macro_f1:.4f}")
                logger.info(f"    Micro F1: {event_micro_f1:.4f}")
                
                results[f'{event_name}_macro_f1'] = event_macro_f1
                results[f'{event_name}_micro_f1'] = event_micro_f1
        
        logger.info("=" * 60)
        
        return results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    event_names: List[str] = None
) -> Dict[str, float]:
    """
    评估模型的便捷函数
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 运行设备
        event_names: 事件名称列表
        
    Returns:
        评估结果字典
    """
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        event_names=event_names
    )
    
    return evaluator.evaluate()

