"""
配置文件
用于管理LLM调用的相关配置参数
"""
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    """LLM配置类"""
    # API配置
    api_key: str = ""
    base_url: str = ""
    
    # 模型配置
    model_name: str = "gpt-3.5-turbo-16k-0613"
    max_tokens: int = 1450
    
    # 重试配置
    max_retry_attempts: int = 6
    min_wait_seconds: int = 1
    max_wait_seconds: int = 2
    
    # 数据文件路径
    data_file: str = "data.xlsx"
    output_file: str = "output.txt"


@dataclass
class GLMConfig:
    """ChatGLM配置类"""
    # 模型路径
    model_path: str = "/THUDM/chatglm2-6b/"
    
    # 数据文件路径
    data_file: str = "data.xlsx"
    output_file: str = "output.txt"
    
    # 设备配置
    device: str = "cuda"
    use_half_precision: bool = True


def load_config_from_env() -> LLMConfig:
    """从环境变量加载配置"""
    return LLMConfig(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", ""),
        data_file=os.getenv("DATA_FILE", "data.xlsx"),
        output_file=os.getenv("OUTPUT_FILE", "output.txt")
    )

