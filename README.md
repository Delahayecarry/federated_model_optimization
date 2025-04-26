# 联邦学习模型优化项目

这个项目使用TensorFlow Federated (TFF)框架实现了联邦学习模型，用于分布式环境下的能效预测。

## 项目结构

```
federated_model_optimization/
│
├── data/                   # 数据文件目录
│   └── Office_Cold_6000-9000_6-10_13.npz  # 示例数据集
│
├── models/                 # 保存训练模型的目录
│   ├── federated_model/    # SavedModel格式模型
│   └── federated_model.h5  # H5格式模型
│
├── src/                    # 源代码
│   ├── config.py           # 配置参数
│   ├── data_utils.py       # 数据处理工具
│   ├── evaluation.py       # 模型评估
│   ├── federated_training.py  # 联邦训练实现
│   ├── main.py             # 主程序入口
│   └── model_definition.py # 模型定义
│
├── .gitignore              # Git忽略文件
├── pyproject.toml          # 项目配置和依赖
└── README.md               # 项目说明文档
```

## 环境要求

- Python 3.10+
- TensorFlow 2.14+
- TensorFlow Federated 0.87.0+

## 安装依赖

```bash
# 使用pip
pip install -r requirements.txt

# 或使用uv
uv pip install -e .
```

## 使用方法

### 训练模型

```bash
# 基本训练
python -m src.main train --data_path=./data/Office_Cold_6000-9000_6-10_13.npz

# 自定义参数
python -m src.main train --data_path=./data/Office_Cold_6000-9000_6-10_13.npz --num_rounds=100 --client_lr=0.01 --server_lr=1.0
```

### 评估模型

```bash
python -m src.main evaluate --model_path=./models/federated_model.h5
```

### 超参数优化

```bash
python -m src.main hypertune --search_layers="3,5,7" --search_neurons="5,10,20" --search_activations="relu,tanh"
```

## 项目特点

- 实现了联邦学习模型训练与评估
- 支持早停机制以避免过拟合
- 优化的日志输出，减少警告和冗余信息
- 保存模型为多种格式(SavedModel和H5)

## 联邦学习参数

- 客户端学习率: 0.02 (可配置)
- 服务器学习率: 1.0 (可配置)
- 训练轮数: 50 (可配置)
- 早停阈值: 0.005 (可配置)

## 贡献指南

欢迎贡献代码或提出问题。请遵循以下步骤：

1. Fork项目
2. 创建您的功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request
