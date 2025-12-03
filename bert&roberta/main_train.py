"""
训练脚本
使用方法:
    python main_train.py --model_name bert --num_epochs 50 --train_batch_size 16
"""
import torch
from transformers import BertTokenizer

from config import get_config_from_args
from model import create_model
from data_loader import create_data_loader
from trainer import train_model
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
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {config.pretrained_weights}")
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_weights)
    
    # 创建数据加载器
    logger.info("创建训练数据加载器...")
    train_loader = create_data_loader(
        data_path=config.train_file,
        tokenizer=tokenizer,
        model_name=config.model_name,
        batch_size=config.train_batch_size,
        max_length=config.max_length,
        shuffle=True,
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
    
    # 训练模型
    logger.info("开始训练...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        device=config.device,
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_training_steps,
        save_path=config.save_file
    )
    
    logger.info("训练完成!")
    logger.info(f"最终损失: {history['loss'][-1]:.4f}")
    logger.info(f"最终准确率: {history['accuracy'][-1]:.4f}")


if __name__ == '__main__':
    main()

