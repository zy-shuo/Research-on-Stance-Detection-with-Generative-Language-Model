"""
T5测试脚本
使用方法:
    python main_test.py --weights_name mt5model-Feb16_11-59-53-epoch4.pth
"""
import os
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from config import get_config_from_args
from data_utils import load_json
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
    
    # 检查是否指定了权重文件
    if not hasattr(config, 'weights_name') or not config.weights_name:
        logger.error("请使用 --weights_name 参数指定要加载的权重文件")
        return
    
    logger.info("配置信息:")
    print(config)
    
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
    
    # 创建测试数据加载器
    logger.info("创建测试数据加载器...")
    test_loader = create_data_loader(
        data_path=data_files['test'],
        tokenizer=tokenizer,
        classes_map=classes_map,
        prefix_text=config.prefix_text,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    # 加载模型配置和权重
    logger.info(f"加载T5模型配置: {config.pretrained_model_name_or_path}")
    model_config = T5Config.from_pretrained(config.pretrained_model_name_or_path)
    
    weights_path = os.path.join(config.save_weights_path, config.weights_name)
    logger.info(f"加载模型权重: {weights_path}")
    
    model = T5ForConditionalGeneration.from_pretrained(
        weights_path,
        config=model_config
    )
    model.to(config.device)
    
    # 评估
    logger.info("开始评估...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        label_id_list=label_id_list,
        device=config.device
    )
    
    logger.info("评估完成!")
    logger.info(f"最终结果: {results}")


if __name__ == "__main__":
    main()

