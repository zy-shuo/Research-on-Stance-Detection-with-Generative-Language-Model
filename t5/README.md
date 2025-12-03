# T5 立场检测模块

本模块使用T5（Text-to-Text Transfer Transformer）模型进行立场检测任务。T5将所有NLP任务统一为文本生成任务，通过生成立场标签来实现分类。

## 目录结构

```
t5/
├── config.py           # 配置管理
├── data_utils.py       # 数据工具函数
├── data_loader.py      # 数据加载
├── trainer.py          # 训练模块
├── evaluator.py        # 评估模块
├── main_train.py       # 训练脚本
├── main_test.py        # 测试脚本
├── classes_map.json    # 类别映射文件
└── README.md           # 本文档
```

## 安装依赖

```bash
pip install torch transformers scikit-learn tqdm
```

## 类别映射文件

`classes_map.json` 定义了立场标签到数字的映射：

```json
{
    "favor": 0,
    "against": 1,
    "neutral": 2
}
```

## 使用方法

### 1. 准备数据

数据应放在 `dataset/` 目录下，包含三个文件：
- `train.json`: 训练集
- `validation.json`: 验证集
- `test.json`: 测试集

每条数据格式：
```json
{
    "label": 0,
    "text": "评论文本",
    "target": "目标事件",
    "简约背景": "背景信息（可选）",
    "全部立场标签": "标签信息（可选）"
}
```

### 2. 训练模型

#### 使用默认配置

```bash
python main_train.py
```

#### 自定义配置

```bash
python main_train.py \
    --pretrained_model_name_or_path ./mt5model \
    --num_train_epochs 12 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --use_adafactor \
    --use_weighted_sampler
```

### 3. 测试模型

#### 使用命令行

```bash
python main_test.py \
    --weights_name mt5model-Feb16_11-59-53-epoch4.pth \
    --pretrained_model_name_or_path ./mt5model
```

## 配置参数说明

### 模型参数
- `pretrained_model_name_or_path`: 预训练T5模型路径，默认'./mt5model'
- `classes_map_path`: 类别映射文件路径，默认'classes_map.json'
- `prefix_text`: 输入文本前缀，默认"判断评论立场："

### 训练参数
- `num_train_epochs`: 训练轮数，默认12
- `batch_size`: 批次大小，默认4
- `learning_rate`: 学习率，默认2e-4
- `weight_decay`: 权重衰减，默认0.0
- `lr_warmup_steps`: 学习率预热步数，默认0

### 优化器配置
- `use_adafactor`: 是否使用Adafactor优化器，默认True
- `use_adafactor_schedule`: 是否使用Adafactor自适应学习率，默认True

### 数据参数
- `data_dir`: 数据目录，默认'dataset'
- `use_weighted_sampler`: 是否使用加权采样，默认False

### 环境参数
- `device`: 运行设备，自动检测CUDA
- `num_workers`: 数据加载线程数，默认0
- `save_weights_path`: 模型权重保存目录，默认'weights'
- `seed`: 随机种子，默认42

## 评估指标

- **准确率 (Accuracy)**: 预测正确的样本比例
- **宏平均F1 (Macro F1)**: 各类别F1分数的平均值
- **微平均F1 (Micro F1)**: 全局计算的F1分数
- **加权F1 (Weighted F1)**: 按类别样本数加权的F1分数
- **各类别F1**: 每个类别单独的F1分数

