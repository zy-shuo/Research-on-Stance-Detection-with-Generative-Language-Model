"""
ChatGLM客户端模块
用于调用本地ChatGLM模型进行立场检测
"""
import logging
from typing import List, Optional
from pathlib import Path

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from config import GLMConfig
from prompt_templates import build_prompt


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GLMClient:
    """ChatGLM客户端类"""
    
    def __init__(self, config: Optional[GLMConfig] = None):
        """
        初始化ChatGLM客户端
        
        Args:
            config: GLM配置对象
        """
        self.config = config or GLMConfig()
        
        # 检查模型路径
        if not Path(self.config.model_path).exists():
            raise ValueError(f"模型路径不存在: {self.config.model_path}")
        
        # 加载tokenizer和模型
        logger.info(f"正在加载ChatGLM模型: {self.config.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModel.from_pretrained(
            self.config.model_path,
            trust_remote_code=True
        )
        
        # 配置设备和精度
        if self.config.use_half_precision and self.config.device == "cuda":
            self.model = self.model.half()
        
        if torch.cuda.is_available() and self.config.device == "cuda":
            self.model = self.model.cuda()
        else:
            logger.warning("CUDA不可用，使用CPU运行（速度会较慢）")
            self.config.device = "cpu"
        
        self.model = self.model.eval()
        logger.info("ChatGLM模型加载成功")
    
    def ask_glm(self, prompt: str) -> str:
        """
        调用ChatGLM进行推理
        
        Args:
            prompt: 输入的prompt
            
        Returns:
            ChatGLM的回复文本
        """
        try:
            response, history = self.model.chat(
                self.tokenizer,
                prompt,
                history=[]
            )
            
            logger.info(f"GLM回复: {response}")
            
            # 移除换行符
            return response.replace("\n", "")
            
        except Exception as e:
            logger.error(f"ChatGLM推理失败: {str(e)}")
            return "推理失败"
    
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
                    
                    # 调用GLM
                    response = self.ask_glm(prompt)
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
    config = GLMConfig(
        model_path="THUDM/chatglm2-6b/",
        data_file="../data/v2/test.xlsx",
        output_file="glm_output.txt"
    )
    
    # 创建客户端
    client = GLMClient(config)
    
    # 读取数据
    df = pd.read_excel(config.data_file)
    
    # 选择prompt模板类型
    # 可选: basic, simple, background, label_constrained, few_shot, chain_of_thought, cross_target
    template_type = "cross_target"
    
    # 处理数据
    results = client.process_dataframe(df, template_type=template_type)
    
    print(f"处理完成，共{len(results)}条结果")


if __name__ == "__main__":
    main()

