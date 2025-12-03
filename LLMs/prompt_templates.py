"""
Prompt模板管理
用于管理不同的prompt策略
"""
from typing import Dict, Any


class PromptTemplate:
    """Prompt模板基类"""
    
    @staticmethod
    def basic_prompt(target: str, text: str) -> str:
        """
        基础prompt模板
        
        Args:
            target: 目标事件
            text: 评论文本
            
        Returns:
            构建好的prompt字符串
        """
        return f'针对目标有一评论，请判断评论对目标的立场，不必解释原因，回答的形式为立场：[支持/反对/中立]。\n目标：{target}\n评论：{text}'
    
    @staticmethod
    def simple_prompt(target: str, text: str) -> str:
        """简洁prompt模板"""
        return f'请判断评论"{text}"对"{target}"事件的立场是什么，不必解释原因，直接回答"支持"、"反对"或"中立"。'
    
    @staticmethod
    def background_prompt(target: str, text: str, background: str) -> str:
        """
        带背景信息的prompt模板
        
        Args:
            target: 目标事件
            text: 评论文本
            background: 背景信息
            
        Returns:
            构建好的prompt字符串
        """
        return f'针对目标有一评论，请判断评论对目标的立场，不必解释原因，回答的形式为立场：[支持/反对/中立]。\n背景：{background}\n目标：{target}\n评论：{text}'
    
    @staticmethod
    def label_constrained_prompt(target: str, text: str, label_options: str) -> str:
        """
        带标签约束的prompt模板
        
        Args:
            target: 目标事件
            text: 评论文本
            label_options: 可选标签选项
            
        Returns:
            构建好的prompt字符串
        """
        return f'针对目标有一评论，请判断评论对目标的立场，回答的形式为立场：{label_options}。\n目标：{target}\n评论：{text}'
    
    @staticmethod
    def few_shot_prompt(target: str, text: str, example: str, example_label: str) -> str:
        """
        Few-shot学习prompt模板
        
        Args:
            target: 目标事件
            text: 评论文本
            example: 示例评论
            example_label: 示例标签
            
        Returns:
            构建好的prompt字符串
        """
        return f'针对目标有一评论，请判断评论b对目标的立场，不必解释原因，回答的形式为立场：[支持/反对/中立]。\n目标：{target}\n评论a：{example}\n立场a：{example_label}\n评论b：{text}'
    
    @staticmethod
    def chain_of_thought_prompt(target: str, text: str, example: str, reasoning: str) -> str:
        """
        思维链prompt模板
        
        Args:
            target: 目标事件
            text: 评论文本
            example: 示例评论
            reasoning: 推理过程
            
        Returns:
            构建好的prompt字符串
        """
        return f'针对目标有一评论，请判断评论b对目标的立场，回答的形式为立场：[支持/反对/中立]。\n目标：{target}\n评论a：{example}\n立场a：让我们一步一步来思考。{reasoning}\n评论b：{text}\n立场：让我们一步一步来思考。'
    
    @staticmethod
    def cross_target_prompt(target_a: str, text_a: str, target_b: str, text_b: str) -> str:
        """
        跨目标prompt模板
        
        Args:
            target_a: 目标事件A
            text_a: 评论文本A
            target_b: 目标事件B
            text_b: 评论文本B
            
        Returns:
            构建好的prompt字符串
        """
        return f'针对目标b有一评论，请判断评论b对目标b的立场，回答的形式为立场：[支持/反对/中立]。\n目标a：{target_a}\n评论a：{text_a}\n目标b：{target_b}\n评论b：{text_b}'


def build_prompt(data_row: Dict[str, Any], template_type: str = "basic") -> str:
    """
    根据数据行和模板类型构建prompt
    
    Args:
        data_row: 数据行字典
        template_type: 模板类型
        
    Returns:
        构建好的prompt字符串
        
    Raises:
        ValueError: 当模板类型不支持时抛出
    """
    target = data_row.get('target', '')
    text = data_row.get('text', '')
    
    if template_type == "basic":
        return PromptTemplate.basic_prompt(target, text)
    elif template_type == "simple":
        return PromptTemplate.simple_prompt(target, text)
    elif template_type == "background":
        background = data_row.get('简约背景', '')
        return PromptTemplate.background_prompt(target, text, background)
    elif template_type == "label_constrained":
        label_options = data_row.get('全部立场标签', '')
        return PromptTemplate.label_constrained_prompt(target, text, label_options)
    elif template_type == "few_shot":
        example = data_row.get('示例', '')
        example_label = data_row.get('示例立场', '')
        return PromptTemplate.few_shot_prompt(target, text, example, example_label)
    elif template_type == "chain_of_thought":
        example = data_row.get('示例', '')
        reasoning = data_row.get('思维链', '')
        return PromptTemplate.chain_of_thought_prompt(target, text, example, reasoning)
    elif template_type == "cross_target":
        target_a = data_row.get('示例目标', '')
        text_a = data_row.get('示例2', '')
        return PromptTemplate.cross_target_prompt(target_a, text_a, target, text)
    else:
        raise ValueError(f"不支持的模板类型: {template_type}")

