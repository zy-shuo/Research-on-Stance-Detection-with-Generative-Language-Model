# BERT/RoBERTa 立场检测模块

本模块使用BERT和RoBERTa预训练模型进行微调，实现立场检测任务。

## 目录结构

```
bert&roberta/
├── config.py           # 配置管理
├── model.py           # 模型定义
├── data_loader.py     # 数据加载
├── trainer.py         # 训练模块
├── evaluator.py       # 评估模块
├── main_train.py      # 训练脚本
├── main_test.py       # 测试脚本
└── README.md          # 本文档
```

## 安装依赖

```bash
pip install torch transformers scikit-learn tqdm
```

## 使用方法

### 1. 训练模型

#### 使用默认配置

```bash
python main_train.py
```

#### 自定义配置

```bash
python main_train.py \
    --model_name bert \
    --num_epochs 50 \
    --train_batch_size 16 \
    --lr 4e-5 \
    --pretrained_weights ../model \
    --train_file ../data/v2/train.json \
    --save_file bert_model.params
```

## 配置参数说明

### 模型参数
- `model_name`: 模型名称，'bert'或'roberta'
- `num_classes`: 分类类别数，默认3（支持/反对/中立）
- `pretrained_weights`: 预训练模型路径

### 训练参数
- `train_batch_size`: 训练批次大小，默认16
- `test_batch_size`: 测试批次大小，默认32
- `num_epochs`: 训练轮数，默认50
- `learning_rate`: 学习率，默认4e-5
- `weight_decay`: 权重衰减，默认0.01
- `warmup_steps`: 学习率预热步数，默认25
- `total_training_steps`: 总训练步数，默认40

### 数据参数
- `max_length`: 最大序列长度，默认500
- `train_file`: 训练数据文件路径
- `test_file`: 测试数据文件路径

### 环境参数
- `device`: 运行设备，自动检测CUDA
- `num_workers`: 数据加载线程数，默认0
- `save_file`: 模型保存文件路径
- `seed`: 随机种子，默认42

## 数据格式

输入数据应为JSON格式，每条数据包含：

```json
{
    "label": 0,
    "text": "评论文本",
    "target": "目标事件",
    "简约背景": "背景信息（可选）",
    "全部立场标签": "标签信息（可选）"
}
```

标签说明：
- 0: 支持 (favor)
- 1: 反对 (against)
- 2: 中立 (neutral)

## 模型架构

### BERT模型
- 基于预训练BERT
- 冻结底部8层，微调顶部4层和pooler层
- 添加Dropout和全连接分类层
- 输出经过softmax归一化

### RoBERTa模型
- 基于预训练RoBERTa
- 冻结底部6层，微调顶部6层和pooler层
- 其他架构与BERT相同

## 评估指标

### 整体指标
- **准确率 (Accuracy)**: 预测正确的样本比例
- **宏平均F1 (Macro F1)**: 各类别F1分数的平均值
- **微平均F1 (Micro F1)**: 全局计算的F1分数
- **各类别F1**: 支持、反对、中立三个类别的F1分数

### 分事件指标
对五个事件分别计算Macro F1和Micro F1：
1. 恶意殴打他人者的妻女被网暴
2. 女子不让6岁男童上女厕所遭痛骂
3. 警方通告胡鑫宇为自杀
4. 满江红起诉大V
5. 泼水节女生选择原谅对方

## 注意事项

1. 需要预先下载BERT/RoBERTa预训练模型
2. 建议使用GPU训练，CPU训练会很慢
3. 确保数据格式正确
4. 训练前检查配置参数
5. 定期保存模型检查点

## 常见问题

### Q: 如何使用自己的预训练模型？
A: 修改`pretrained_weights`参数指向你的模型路径。

### Q: 如何调整冻结的层数？
A: 在`model.py`中修改`freeze_layers`参数。

### Q: 如何添加背景信息？
A: 在`data_loader.py`的`create_data_loader`函数中设置`use_background=True`。

### Q: 训练时显存不足怎么办？
A: 减小`train_batch_size`或`max_length`参数。

