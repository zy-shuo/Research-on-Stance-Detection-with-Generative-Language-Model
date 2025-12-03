"""
T5模型评估模块
"""
import sys
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score, f1_score
from typing import Dict, List, Tuple
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5Evaluator:
    """T5评估器"""
    
    def __init__(
        self,
        model: T5ForConditionalGeneration,
        test_loader: DataLoader,
        label_id_list: List[int],
        device: torch.device,
        event_names: List[str] = None
    ):
        """
        初始化评估器
        
        Args:
            model: T5模型
            test_loader: 测试数据加载器
            label_id_list: 标签ID列表
            device: 运行设备
            event_names: 事件名称列表
        """
        self.model = model
        self.test_loader = test_loader
        self.label_id_list = label_id_list
        self.device = device
        self.event_names = event_names or [
            "恶意殴打他人者的妻女被网暴",
            "女子不让6岁男童上女厕所遭痛骂",
            "警方通告胡鑫宇为自杀",
            "满江红起诉大V",
            "泼水节女生选择原谅对方"
        ]
        
        logger.info("评估器初始化完成")
    
    @torch.no_grad()
    def predict(self) -> Tuple[List[int], List[int]]:
        """
        对测试集进行预测
        
        Returns:
            (预测标签列表, 真实标签列表)
        """
        self.model.eval()
        
        predicted_labels = []
        ground_truth_labels = []
        
        logger.info("开始预测...")
        
        pbar = tqdm(self.test_loader, desc="[测试]", file=sys.stdout)
        
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
            
            pred_labels = outputs[:, 1]  # 取第二个token
            
            # 检查是否有标签外的预测
            for pred in pred_labels:
                if pred.item() not in self.label_id_list:
                    logger.warning(f"预测标签不在标签列表中: {pred.item()}")
            
            # 收集标签
            ground_truth_labels.extend(labels[:, 0].cpu().tolist())
            predicted_labels.extend(pred_labels.cpu().tolist())
            
            # 更新进度条
            if (step + 1) % 10 == 0:
                accuracy = accuracy_score(ground_truth_labels, predicted_labels)
                pbar.set_postfix({'acc': f'{accuracy:.4f}'})
        
        logger.info(f"预测完成，共{len(predicted_labels)}条数据")
        
        return predicted_labels, ground_truth_labels
    
    def evaluate(self) -> Dict[str, float]:
        """
        评估模型性能
        
        Returns:
            包含各项指标的字典
        """
        predictions, labels = self.predict()
        
        # 计算整体指标
        accuracy = accuracy_score(labels, predictions)
        acc_nums = accuracy_score(labels, predictions, normalize=False)
        macro_f1 = f1_score(labels, predictions, average='macro')
        micro_f1 = f1_score(labels, predictions, average='micro')
        weighted_f1 = f1_score(labels, predictions, average='weighted')
        
        # 计算每个类别的F1分数
        unique_labels = sorted(set(labels))
        f1_per_class = {}
        for label_id in unique_labels:
            f1 = f1_score(labels, predictions, labels=[label_id], average='micro')
            f1_per_class[label_id] = f1
        
        results = {
            'accuracy': accuracy,
            'acc_nums': acc_nums,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1
        }
        
        # 添加每个类别的F1
        for label_id, f1 in f1_per_class.items():
            results[f'f1_class_{label_id}'] = f1
        
        # 打印整体结果
        logger.info("=" * 80)
        logger.info("整体评估结果:")
        logger.info(f"  准确率 (Accuracy): {accuracy:.4f}")
        logger.info(f"  正确数量: {acc_nums}")
        logger.info(f"  宏平均F1 (Macro F1): {macro_f1:.4f}")
        logger.info(f"  微平均F1 (Micro F1): {micro_f1:.4f}")
        logger.info(f"  加权F1 (Weighted F1): {weighted_f1:.4f}")
        
        for label_id, f1 in f1_per_class.items():
            logger.info(f"  类别{label_id} F1: {f1:.4f}")
        
        # 计算分事件指标
        if len(predictions) == 500:  # 假设每个事件100条数据
            logger.info("=" * 80)
            logger.info("分事件评估结果:")
            
            for i, event_name in enumerate(self.event_names):
                start_idx = i * 100
                end_idx = (i + 1) * 100
                
                event_predictions = predictions[start_idx:end_idx]
                event_labels = labels[start_idx:end_idx]
                
                event_macro_f1 = f1_score(
                    event_labels,
                    event_predictions,
                    average='macro'
                )
                event_micro_f1 = f1_score(
                    event_labels,
                    event_predictions,
                    average='micro'
                )
                
                logger.info(f"  {event_name}:")
                logger.info(f"    Macro F1: {event_macro_f1:.4f}")
                logger.info(f"    Micro F1: {event_micro_f1:.4f}")
                
                results[f'{event_name}_macro_f1'] = event_macro_f1
                results[f'{event_name}_micro_f1'] = event_micro_f1
        
        logger.info("=" * 80)
        
        return results


def evaluate_model(
    model: T5ForConditionalGeneration,
    test_loader: DataLoader,
    label_id_list: List[int],
    device: torch.device,
    event_names: List[str] = None
) -> Dict[str, float]:
    """
    评估模型的便捷函数
    
    Args:
        model: T5模型
        test_loader: 测试数据加载器
        label_id_list: 标签ID列表
        device: 运行设备
        event_names: 事件名称列表
        
    Returns:
        评估结果字典
    """
    evaluator = T5Evaluator(
        model=model,
        test_loader=test_loader,
        label_id_list=label_id_list,
        device=device,
        event_names=event_names
    )
    
    return evaluator.evaluate()

