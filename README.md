# 基于联邦学习的预测模型优化

本项目实现了一个基于TensorFlow Federated (TFF)的联邦学习框架，用于优化建筑能耗预测模型。通过联邦学习，可以在保护数据隐私的前提下，利用多个客户端（建筑）的数据进行模型训练。

## 项目结构

```
federated_model_optimization/
├── src/                      # 源代码目录
│   ├── __init__.py           # 包初始化文件
│   ├── main.py               # 主入口点
│   ├── config.py             # 配置参数
│   ├── data_utils.py         # 数据处理工具
│   ├── model_definition.py   # 模型定义
│   ├── federated_training.py # 联邦训练逻辑
│   └── evalution.py          # 模型评估（注意：文件名有拼写错误）
├── data/                     # 数据目录（需要自行添加数据）
├── models/                   # 保存模型的目录
├── pyproject.toml            # 项目配置和依赖
└── README.md                 # 项目说明文档
```

## 功能特点

- **联邦学习框架**: 使用TFF构建的联邦学习系统，支持多客户端数据协作训练
- **模型优化**: 支持超参数优化，自动搜索最佳模型配置
- **灵活配置**: 通过命令行参数或配置文件灵活调整模型和训练参数
- **完整评估**: 提供多种评估指标，包括MSE、MAE和CVRMSE

## 依赖环境

本项目使用`uv`进行依赖管理。主要依赖包括：

- Python = 3.8.20
- TensorFlow >= 2.10.0
- TensorFlow Federated >= 0.20.0
- NumPy >= 1.23.5
- Pandas >= 1.5.3

## 安装说明

1. 克隆仓库：
   ```bash
   git clone <仓库地址>
   cd federated_model_optimization
   ```

2. 使用uv创建虚拟环境并安装依赖：
   ```bash
   uv venv
   uv pip install -e .
   ```

## 使用方法

项目支持三种运行模式：

### 1. 训练模式

```bash
python -m src.main train --data_path=<数据路径> --num_layers=5 --num_neurons=10 --activation=tanh
```

### 2. 评估模式

```bash
python -m src.main evaluate --model_path=<模型路径> --data_path=<测试数据路径>
```

### 3. 超参数优化模式

```bash
python -m src.main hypertune --data_path=<数据路径> --search_layers="1,3,5" --search_neurons="5,10,15,20" --search_activations="relu,tanh,sigmoid"
```

## 数据说明

项目需要`.npz`格式的数据文件，包含以下键：
- `data`: 原始未归一化数据
- `data_norm`: 归一化后的数据
- `name`: 客户端（建筑）标识符

您需要将数据文件放在`data/`目录下，或在运行时通过`--data_path`参数指定数据路径。

## 配置调整

1. 修改`src/config.py`中的默认参数
2. 通过命令行参数覆盖默认配置

## 从Notebook迁移

本项目是从Jupyter Notebook迁移而来，保留了原有的核心功能和模型结构，同时进行了以下改进：

1. 模块化设计：将代码分解为逻辑清晰的模块
2. 命令行接口：添加了命令行参数解析，便于灵活运行
3. 配置集中化：使用专门的配置模块管理所有参数
4. 错误处理增强：添加了更多异常处理和友好的错误消息
5. 日志系统：实现了完整的日志记录系统
