"""
T5数据处理工具
"""
import json
import logging
from typing import List, Dict, Any
from pathlib import Path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(file_path: str) -> Any:
    """
    加载JSON文件
    
    Args:
        file_path: JSON文件路径
        
    Returns:
        解析后的数据
        
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载JSON文件: {file_path}")
        
        # 如果是列表，显示数量
        if isinstance(data, list):
            logger.info(f"  数据条数: {len(data)}")
        
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON格式错误: {file_path}")
        raise
    except Exception as e:
        logger.error(f"加载文件失败: {file_path}, 错误: {str(e)}")
        raise


def save_json(data: Any, file_path: str, indent: int = 2):
    """
    保存数据为JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 保存路径
        indent: 缩进空格数
    """
    file_path = Path(file_path)
    
    # 创建父目录
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        
        logger.info(f"成功保存JSON文件: {file_path}")
        
    except Exception as e:
        logger.error(f"保存文件失败: {file_path}, 错误: {str(e)}")
        raise


def validate_data_format(data: List[Dict[str, Any]]) -> bool:
    """
    验证数据格式是否正确
    
    Args:
        data: 数据列表
        
    Returns:
        是否格式正确
    """
    if not isinstance(data, list):
        logger.error("数据格式错误: 应为列表")
        return False
    
    if len(data) == 0:
        logger.warning("数据为空")
        return True
    
    required_fields = ['label', 'text', 'target']
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.error(f"第{i}条数据格式错误: 应为字典")
            return False
        
        for field in required_fields:
            if field not in item:
                logger.error(f"第{i}条数据缺少必需字段: {field}")
                return False
    
    logger.info("数据格式验证通过")
    return True


def get_label_distribution(data: List[Dict[str, Any]]) -> Dict[int, int]:
    """
    统计标签分布
    
    Args:
        data: 数据列表
        
    Returns:
        标签分布字典 {label: count}
    """
    distribution = {}
    
    for item in data:
        label = item['label']
        distribution[label] = distribution.get(label, 0) + 1
    
    logger.info("标签分布:")
    for label, count in sorted(distribution.items()):
        logger.info(f"  标签 {label}: {count} 条 ({100 * count / len(data):.2f}%)")
    
    return distribution

