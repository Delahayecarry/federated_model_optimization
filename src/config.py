# src/config.py

# --- 数据参数 ---
# 默认使用相对路径，便于本地开发和测试
DATA_PATH = './data/Office_Cold_6000-9000_6-10_13.npz'  # 默认相对路径，可通过命令行参数修改
NUM_EXAMPLES_PER_USER = 10000 # 每个客户端最大加载的样本数量
BATCH_SIZE = 500             # 联邦训练/评估的批量大小

# 注意: EC_MIN、EC_MAX、EC_MEAN是从特定数据集派生的。
# 它们用于后续对预测结果进行反归一化。
# 最好在data_utils.py加载数据后计算这些值。
EC_MIN = None # 占位符，将从数据中计算
EC_MAX = None # 占位符，将从数据中计算
EC_MEAN = None # 占位符，将从数据中计算


# --- 模型参数 ---
INPUT_SHAPE = (6,)            # 输入特征的形状

# 模型超参数（设置为notebook中找到的"最佳"值）
# 你可能需要从单独的超参数配置文件或使用命令行参数加载这些参数
NUM_LAYERS = 5                # 隐藏层数量
NUM_NEURONS = 10              # 每层神经元数量
ACTIVATION = 'tanh'           # 激活函数


# --- 训练参数 ---
CLIENT_LEARNING_RATE = 0.02
SERVER_LEARNING_RATE = 1.0
NUM_ROUNDS = 50               # 联邦训练轮数
EARLY_STOPPING_THRESHOLD = 0.005 # 早停阈值


# --- 本地微调参数 (可选，如果实现本地训练部分) ---
LOCAL_BATCH_SIZE = 64
LOCAL_EPOCHS = 100
LOCAL_VALIDATION_SPLIT = 0.2
LOCAL_EARLY_STOPPING_PATIENCE = 5


# --- 输出路径 ---
# 使用相对路径方便不同环境中的部署
MODEL_SAVE_BASE_DIR = './models/' # 保存模型的基础目录

FEDERATED_MODEL_SAVE_NAME = 'federated_model'
LOCAL_MODEL_SAVE_NAME = 'local_model'

FEDERATED_MODEL_H5_SAVE_NAME = 'federated_model.h5'
LOCAL_MODEL_H5_SAVE_NAME = 'local_model.h5'

# 构建完整路径的示例（可以在这里或需要时构建）
# FEDERATED_MODEL_SAVE_PATH = f"{MODEL_SAVE_BASE_DIR}{FEDERATED_MODEL_SAVE_NAME}"
# LOCAL_MODEL_SAVE_PATH = f"{MODEL_SAVE_BASE_DIR}{LOCAL_MODEL_SAVE_NAME}"
# FEDERATED_MODEL_H5_PATH = f"{MODEL_SAVE_BASE_DIR}{FEDERATED_MODEL_H5_SAVE_NAME}"
# LOCAL_MODEL_H5_PATH = f"{MODEL_SAVE_BASE_DIR}{LOCAL_MODEL_H5_SAVE_NAME}"
