# LLMs 立场检测模块

本模块提供了使用大语言模型（GPT和ChatGLM）进行立场检测的功能。

## 目录结构

```
LLMs/
├── config.py              # 配置管理
├── prompt_templates.py    # Prompt模板管理
├── gpt_client.py         # GPT客户端
├── glm_client.py         # ChatGLM客户端
└── README.md             # 本文档
```

## 功能特性

- **多种Prompt模板**: 支持7种不同的prompt策略
  - 基础模板 (basic)
  - 简洁模板 (simple)
  - 带背景信息 (background)
  - 标签约束 (label_constrained)
  - Few-shot学习 (few_shot)
  - 思维链 (chain_of_thought)
  - 跨目标 (cross_target)

## 使用方法

### 1. GPT客户端

```python
from gpt_client import GPTClient
from config import LLMConfig
import pandas as pd

# 创建配置
config = LLMConfig(
    api_key="your-api-key",
    base_url="",  # 可选，使用代理时填写
    data_file="data.xlsx",
    output_file="output.txt"
)

# 创建客户端
client = GPTClient(config)

# 读取数据
df = pd.read_excel(config.data_file)

# 处理数据
results = client.process_dataframe(df, template_type="basic")
```

### 2. ChatGLM客户端

```python
from glm_client import GLMClient
from config import GLMConfig
import pandas as pd

# 创建配置
config = GLMConfig(
    model_path="/path/to/chatglm2-6b/",
    data_file="data.xlsx",
    output_file="output.txt"
)

# 创建客户端
client = GLMClient(config)

# 读取数据
df = pd.read_excel(config.data_file)

# 处理数据
results = client.process_dataframe(df, template_type="basic")
```

### 3. 环境变量配置

GPT客户端支持从环境变量加载配置：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export DATA_FILE="data.xlsx"
export OUTPUT_FILE="output.txt"
```

然后在代码中：

```python
from gpt_client import GPTClient

# 自动从环境变量加载配置
client = GPTClient()
```

## Prompt模板说明

### 1. 基础模板 (basic)
```
针对目标有一评论，请判断评论对目标的立场，不必解释原因，回答的形式为立场：[支持/反对/中立]。
目标：{target}
评论：{text}
```

### 2. 简洁模板 (simple)
```
请判断评论"{text}"对"{target}"事件的立场是什么，不必解释原因，直接回答"支持"、"反对"或"中立"。
```

### 3. 带背景信息 (background)
在基础模板上添加背景信息，帮助模型更好地理解上下文。

### 4. 标签约束 (label_constrained)
提供具体的标签选项，约束模型的输出范围。

### 5. Few-shot学习 (few_shot)
提供示例评论和标签，让模型学习推理模式。

### 6. 思维链 (chain_of_thought)
引导模型进行逐步推理，提高复杂场景下的准确性。

### 7. 跨目标 (cross_target)
通过对比不同目标的评论，帮助模型理解立场的相对性。

## 配置参数说明

### LLMConfig (GPT)
- `api_key`: OpenAI API密钥
- `base_url`: API基础URL（使用代理时需要）
- `model_name`: 模型名称，默认为"gpt-3.5-turbo-16k-0613"
- `max_tokens`: 最大token数，默认1450
- `max_retry_attempts`: 最大重试次数，默认6
- `data_file`: 输入数据文件路径
- `output_file`: 输出结果文件路径

### GLMConfig (ChatGLM)
- `model_path`: ChatGLM模型路径
- `data_file`: 输入数据文件路径
- `output_file`: 输出结果文件路径
- `device`: 运行设备，"cuda"或"cpu"
- `use_half_precision`: 是否使用半精度，默认True

## 数据格式要求

输入数据应为Excel或DataFrame格式，必须包含以下字段：
- `target`: 目标事件
- `text`: 评论文本

根据使用的prompt模板，可能还需要：
- `简约背景`: 背景信息
- `明确的立场标签`: 标签选项
- `示例`: 示例评论
- `示例立场`: 示例标签
- `思维链`: 推理过程
- `示例目标`: 示例目标
- `示例2`: 示例评论

## 依赖项

```
openai>=1.0.0
transformers>=4.30.0
torch>=2.0.0
pandas>=1.5.0
tenacity>=8.0.0
```

