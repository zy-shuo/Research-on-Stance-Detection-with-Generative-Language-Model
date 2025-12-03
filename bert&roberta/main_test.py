"""
测试脚本
使用方法:
    python main_test.py --model_name bert --save_file bert.params
"""
import torch
from transformers import BertTokenizer

from config import get_config_from_args
from model import create_model
from data_loader import create_data_loader
from evaluator import evaluate_model
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    # 获取配置
    config = get_config_from_args()
    logger.info("配置信息:")
    print(config)
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {config.pretrained_weights}")
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_weights)
    
    # 创建数据加载器
    logger.info("创建测试数据加载器...")
    test_loader = create_data_loader(
        data_path=config.test_file,
        tokenizer=tokenizer,
        model_name=config.model_name,
        batch_size=config.test_batch_size,
        max_length=config.max_length,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 创建模型
    logger.info(f"创建{config.model_name}模型...")
    model = create_model(
        model_name=config.model_name,
        num_classes=config.num_classes,
        pretrained_weights=config.pretrained_weights,
        device=config.device
    )
    
    # 加载模型权重
    logger.info(f"加载模型权重: {config.save_file}")
    model.load_state_dict(torch.load(config.save_file, map_location=config.device))
    
    # 评估模型
    logger.info("开始评估...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=config.device
    )
    
    logger.info("评估完成!")


if __name__ == '__main__':
    main()

