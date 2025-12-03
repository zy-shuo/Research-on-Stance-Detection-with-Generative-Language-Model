"""
T5训练脚本
使用方法:
    python main_train.py --num_train_epochs 12 --batch_size 4
"""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from config import get_config_from_args
from data_utils import load_json
from data_loader import create_data_loader
from trainer import T5Trainer
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
    
    # 获取数据文件路径
    data_files = config.get_data_files()
    
    # 加载tokenizer
    logger.info(f"加载tokenizer: {config.pretrained_model_name_or_path}")
    tokenizer = T5Tokenizer.from_pretrained(config.pretrained_model_name_or_path)
    
    # 加载类别映射
    logger.info(f"加载类别映射: {config.classes_map_path}")
    classes_map = load_json(config.classes_map_path)
    
    # 计算标签ID列表
    label_id_list = []
    for cls in classes_map.keys():
        labels_id = tokenizer.encode(text=cls, add_special_tokens=False)
        label_id_list.append(labels_id[0])
    logger.info(f"标签ID列表: {label_id_list}")
    
    # 创建训练数据加载器
    logger.info("创建训练数据加载器...")
    train_loader = create_data_loader(
        data_path=data_files['train'],
        tokenizer=tokenizer,
        classes_map=classes_map,
        prefix_text=config.prefix_text,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        use_weighted_sampler=config.use_weighted_sampler
    )
    
    # 创建验证数据加载器
    logger.info("创建验证数据加载器...")
    valid_loader = create_data_loader(
        data_path=data_files['validation'],
        tokenizer=tokenizer,
        classes_map=classes_map,
        prefix_text=config.prefix_text,
        batch_size=1,  # 验证时batch_size=1
        shuffle=False,
        num_workers=0
    )
    
    # 加载模型
    logger.info(f"加载T5模型: {config.pretrained_model_name_or_path}")
    model = T5ForConditionalGeneration.from_pretrained(config.pretrained_model_name_or_path)
    model.to(config.device)
    
    # 创建训练器
    trainer = T5Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        label_id_list=label_id_list,
        device=config.device,
        num_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=config.lr_warmup_steps,
        use_adafactor=config.use_adafactor,
        use_adafactor_schedule=config.use_adafactor_schedule
    )
    
    # 训练
    logger.info("开始训练...")
    best_result = trainer.train(save_path=config.save_weights_path)
    
    logger.info("训练完成!")
    logger.info(f"最佳Epoch: {best_result['best_epoch']}")
    logger.info(f"最佳Macro F1: {best_result['best_macro_f1']:.4f}")


if __name__ == "__main__":
    main()

