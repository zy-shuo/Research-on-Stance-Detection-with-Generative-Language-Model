"""
GPT客户端模块
用于调用OpenAI GPT API进行立场检测
"""
import logging
from typing import List, Optional
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

from config import LLMConfig, load_config_from_env
from prompt_templates import build_prompt


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPTClient:
    """GPT客户端类"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        初始化GPT客户端
        
        Args:
            config: LLM配置对象，如果为None则从环境变量加载
        """
        self.config = config or load_config_from_env()
        
        if not self.config.api_key:
            raise ValueError("API Key未设置，请在config中设置或通过环境变量OPENAI_API_KEY提供")
        
        self.client = OpenAI(
            base_url=self.config.base_url if self.config.base_url else None,
            api_key=self.config.api_key
        )
        
        logger.info("GPT客户端初始化成功")
    
    @retry(
        wait=wait_random_exponential(min=1, max=2),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(Exception)
    )
    def ask_gpt(self, prompt: str) -> str:
        """
        调用GPT API进行推理
        
        Args:
            prompt: 输入的prompt
            
        Returns:
            GPT的回复文本
            
        Raises:
            Exception: 当API调用失败时抛出
        """
        # 检查prompt长度
        if len(prompt) >= self.config.max_tokens:
            logger.warning(f"Prompt长度({len(prompt)})超出限制({self.config.max_tokens})")
            return '超出限制'
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            message = response.choices[0].message.content
            logger.info(f"GPT回复: {message}")
            
            # 移除换行符
            return message.replace("\n", "")
            
        except Exception as e:
            logger.error(f"GPT API调用失败: {str(e)}")
            raise
    
    def process_dataframe(
        self, 
        df: pd.DataFrame, 
        template_type: str = "basic",
        output_file: Optional[str] = None
    ) -> List[str]:
        """
        批量处理DataFrame中的数据
        
        Args:
            df: 包含评论数据的DataFrame
            template_type: prompt模板类型
            output_file: 输出文件路径，如果为None则使用配置中的路径
            
        Returns:
            所有回复的列表
        """
        output_file = output_file or self.config.output_file
        results = []
        
        logger.info(f"开始处理{len(df)}条数据，使用模板类型: {template_type}")
        
        with open(output_file, 'a', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                try:
                    # 构建prompt
                    prompt = build_prompt(row.to_dict(), template_type)
                    
                    # 调用GPT
                    response = self.ask_gpt(prompt)
                    results.append(response)
                    
                    # 写入文件
                    f.write(response + "\n")
                    
                    logger.info(f"处理进度: {idx + 1}/{len(df)}")
                    
                except Exception as e:
                    logger.error(f"处理第{idx}行数据时出错: {str(e)}")
                    results.append("处理失败")
                    f.write("处理失败\n")
        
        logger.info(f"处理完成，结果已保存到: {output_file}")
        return results


def main():
    """主函数"""
    # 创建配置
    config = LLMConfig(
        api_key="your-api-key-here",  # 请替换为实际的API Key
        base_url="",  # 如果使用代理，请填写base_url
        data_file="../data/v2/test.xlsx",
        output_file="gpt_output.txt"
    )
    
    # 创建客户端
    client = GPTClient(config)
    
    # 读取数据
    df = pd.read_excel(config.data_file)
    
    # 选择prompt模板类型
    # 可选: basic, simple, background, label_constrained, few_shot, chain_of_thought, cross_target
    template_type = "basic"
    
    # 处理数据
    results = client.process_dataframe(df, template_type=template_type)
    
    print(f"处理完成，共{len(results)}条结果")


if __name__ == "__main__":
    main()

