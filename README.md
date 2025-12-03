# 基于生成式语言模型的立场检测研究

本项目实现了多种基于生成式语言模型的立场检测方法，包括大语言模型（GPT、ChatGLM）的零样本/少样本学习，以及预训练模型（BERT、RoBERTa、T5）的微调方法。

## 🎯 项目简介

立场检测（Stance Detection）是自然语言处理中的一项重要任务，旨在判断文本对特定目标的态度倾向（支持、反对或中立）。本项目针对中文社交媒体（微博）场景，实现了多种基于生成式语言模型的立场检测方法。

## 📁 项目结构

```
Research-on-Stance-Detection-with-Generative-Language-Model/
├── LLMs/                      # 大语言模型模块
│   ├── config.py             # 配置管理
│   ├── prompt_templates.py   # Prompt模板
│   ├── gpt_client.py         # GPT客户端
│   ├── glm_client.py         # ChatGLM客户端
│   └── README.md             # 模块文档
│
├── bert&roberta/             # BERT/RoBERTa微调模块
│   ├── config.py             # 配置管理
│   ├── model.py              # 模型定义
│   ├── data_loader.py        # 数据加载
│   ├── trainer.py            # 训练模块
│   ├── evaluator.py          # 评估模块
│   ├── main_train.py         # 训练脚本
│   ├── main_test.py          # 测试脚本
│   └── README.md             # 模块文档
│
├── t5/                       # T5微调模块
│   ├── config.py             # 配置管理
│   ├── data_utils.py         # 数据工具
│   ├── data_loader.py        # 数据加载
│   ├── trainer.py            # 训练模块
│   ├── evaluator.py          # 评估模块
│   ├── main_train.py         # 训练脚本
│   ├── main_test.py          # 测试脚本
│   ├── classes_map.json      # 类别映射
│   └── README.md             # 模块文档
│
├── data/                     # 数据集
│   ├── v1/                   # 数据集v1
│   ├── v2/                   # 数据集v2（推荐）
│   └── README.md             # 数据集说明
│
├── requirements.txt          # 依赖列表
└── README.md                 # 本文档
```

## 📊 数据集

### 微博立场检测数据集

本项目使用的是中文微博立场检测数据集，包含5个热点事件：

1. **唐山打人事件** - 恶意殴打他人者的妻女被网暴
2. **胡鑫宇失踪事件** - 警方通告胡鑫宇为自杀
3. **女厕所争议** - 女子不让6岁男童上女厕所遭痛骂
4. **满江红争议** - 满江红起诉大V
5. **泼水节事件** - 女子泼水节被围着泼水撕雨衣

### 数据统计

- **总数据量**: 2500条
- **训练集**: 2000条（每个事件400条）
- **测试集**: 500条（每个事件100条）

### 数据格式

```json
{
    "label": 0,
    "text": "评论文本",
    "target": "目标事件",
    "立场标签": "支持",
    "简约背景": "事件背景简述",
    "全部背景": "事件背景详细描述",
    "明确的立场标签": "()"
}
```

**标签说明**:
- `0`: 支持 (favor)
- `1`: 反对 (against)
- `2`: 中立 (neutral)

详细说明请参考 [data/README.md](data/README.md)

## 🔧 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.0+ (使用GPU时)
- 8GB+ RAM
- 10GB+ 磁盘空间（用于存储预训练模型）

### 安装步骤

1. **克隆项目**

```bash
git clone https://github.com/yourusername/Research-on-Stance-Detection-with-Generative-Language-Model.git
cd Research-on-Stance-Detection-with-Generative-Language-Model
```

2. **创建虚拟环境（推荐）**

```bash
# 使用conda
conda create -n stance python=3.8
conda activate stance

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **下载预训练模型**

- **BERT/RoBERTa**: 从 [Hugging Face](https://huggingface.co/models) 下载中文BERT或RoBERTa模型
- **T5**: 下载 [mT5](https://huggingface.co/google/mt5-base) 或中文T5模型
- **ChatGLM**: 下载 [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b)

将模型放置在相应的目录下，并在配置文件中指定路径。

## 🚀 快速开始

### 1. 使用大语言模型（LLMs）

#### GPT

```python
from LLMs.gpt_client import GPTClient
from LLMs.config import LLMConfig
import pandas as pd

# 配置
config = LLMConfig(
    api_key="your-api-key",
    data_file="data/v2/test.xlsx",
    output_file="gpt_output.txt"
)

# 创建客户端
client = GPTClient(config)

# 读取数据
df = pd.read_excel(config.data_file)

# 处理数据
results = client.process_dataframe(df, template_type="basic")
```

#### ChatGLM

```python
from LLMs.glm_client import GLMClient
from LLMs.config import GLMConfig
import pandas as pd

# 配置
config = GLMConfig(
    model_path="/path/to/chatglm2-6b/",
    data_file="data/v2/test.xlsx"
)

# 创建客户端
client = GLMClient(config)

# 处理数据
df = pd.read_excel(config.data_file)
results = client.process_dataframe(df, template_type="basic")
```

### 2. 微调BERT/RoBERTa

```bash
# 训练
cd bert\&roberta
python main_train.py \
    --model_name bert \
    --num_epochs 50 \
    --train_batch_size 16 \
    --pretrained_weights ../model \
    --train_file ../data/v2/train.json \
    --save_file bert_model.params

# 测试
python main_test.py \
    --model_name bert \
    --save_file bert_model.params \
    --test_file ../data/v2/test.json
```

### 3. 微调T5

```bash
# 训练
cd t5
python main_train.py \
    --pretrained_model_name_or_path ./mt5model \
    --num_train_epochs 12 \
    --batch_size 4

# 测试
python main_test.py \
    --weights_name mt5model-Feb16_11-59-53-epoch4.pth
```
