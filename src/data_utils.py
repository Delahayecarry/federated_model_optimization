import numpy as np
import pandas as pd
import logging

# 使用相对导入访问同一包中的配置
from . import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_data_for_name(data_array, name_array, target_name):
    """
    提取并批处理特定客户端名称的数据。

    参数:
        data_array: 包含特征和标签（最后一列）的numpy数组。
        name_array: 包含与data_array行对应的客户端名称的numpy数组。
        target_name: 要提取数据的特定客户端名称。

    返回:
        一个字典列表，每个字典代表一个批次，
        包含'x'（特征）和'y'（标签）。如果没有该名称的数据，则返回空列表。
    """
    output_sequence = []
    # 查找目标名称的索引
    all_samples_indices = [i for i, n in enumerate(name_array) if n == target_name]

    if not all_samples_indices:
        logger.warning(f"未找到客户端名称的数据: {target_name}")
        return output_sequence

    # 限制每个用户的样本数并创建批次
    num_samples_to_use = min(len(all_samples_indices), config.NUM_EXAMPLES_PER_USER)
    for i in range(0, num_samples_to_use, config.BATCH_SIZE):
        batch_indices = all_samples_indices[i:min(i + config.BATCH_SIZE, num_samples_to_use)]
        if not batch_indices: # 根据范围逻辑不应该发生，但作为保障
             continue
        batch_x = np.array([data_array[idx, :-1] for idx in batch_indices], dtype=np.float32)
        batch_y = np.array([data_array[idx, -1] for idx in batch_indices], dtype=np.float32)
        output_sequence.append({'x': batch_x, 'y': batch_y})

    logger.debug(f"为客户端{target_name}处理了{len(output_sequence)}个批次")
    return output_sequence


def preprocess_and_load_federated_data():
    """
    加载数据集，对其进行联邦学习预处理，
    并计算必要的统计数据。

    返回:
        tuple: 包含以下内容的元组：
            - federated_train_data (list): 训练客户端的数据。
            - federated_valid_data (list): 验证客户端的数据。
            - ec_min (float): 目标变量的最小值。
            - ec_max (float): 目标变量的最大值。
            - ec_mean (float): 目标变量的平均值。
            # - local_data (list): 本地客户端的数据（注释掉，与notebook匹配）
    """
    logger.info(f"从以下位置加载数据: {config.DATA_PATH}")
    try:
        zipfile = np.load(config.DATA_PATH, allow_pickle=True)
    except FileNotFoundError:
        logger.error(f"在{config.DATA_PATH}处未找到数据文件。请检查config.py中的路径。")
        raise
    except Exception as e:
        logger.error(f"加载数据文件时出错: {e}")
        raise

    # 提取数据数组
    # 'data'包含原始的、未归一化的数据
    # 'data_norm'包含归一化的特征+归一化的标签
    # 'name'包含每一行的客户端标识符
    building_data = zipfile['data']
    building_data_norm = zipfile['data_norm']
    building_names = zipfile['name']
    logger.info(f"数据加载成功。形状: {building_data_norm.shape}")

    # 从原始（未归一化）目标变量计算统计数据
    # 假设目标变量是'building_data'中的最后一列
    if building_data.shape[1] < 1:
         raise ValueError("建筑数据数组似乎为空或没有列。")
    target_column = building_data[:, -1]
    ec_min = np.min(target_column)
    ec_max = np.max(target_column)
    ec_mean = np.mean(target_column)
    logger.info(f"计算目标统计数据: Min={ec_min:.4f}, Max={ec_max:.4f}, Mean={ec_mean:.4f}")

    # 获取唯一的客户端名称
    list_of_names = pd.Series(building_names).unique()
    if len(list_of_names) < 3:
         raise ValueError(f"数据集需要至少3个唯一的客户端名称用于训练/验证/本地拆分，找到{len(list_of_names)}")
    logger.info(f"找到{len(list_of_names)}个唯一的客户端名称。")

    # 为训练、验证（和可能的本地测试）拆分客户端名称
    # 使用归一化数据进行联邦处理
    train_client_names = list_of_names[:-2]
    valid_client_names = list_of_names[-2:-1]
    # local_client_names = list_of_names[-1:] # 保持与注释掉的notebook代码一致

    logger.info(f"分配{len(train_client_names)}个客户端用于训练，{len(valid_client_names)}个用于验证。")

    # 通过处理每个客户端名称的数据创建联邦数据集
    federated_train_data = [
        _get_data_for_name(building_data_norm, building_names, name)
        for name in train_client_names
    ]
    # 过滤掉可能没有返回数据的客户端
    federated_train_data = [client_data for client_data in federated_train_data if client_data]

    federated_valid_data = [
        _get_data_for_name(building_data_norm, building_names, name)
        for name in valid_client_names
    ]
    federated_valid_data = [client_data for client_data in federated_valid_data if client_data]

    # --- 注释掉的本地数据处理，映射notebook ---
    # local_data = [
    #     _get_data_for_name(building_data_norm, building_names, name)
    #     for name in local_client_names
    # ]
    # local_data = [client_data for client_data in local_data if client_data]
    #
    # # 如果需要，进一步拆分本地数据（来自notebook的示例）
    # if local_data:
    #     local_train_data = local_data[0][:2] # 示例：前2个批次用于训练
    #     local_test_data = local_data[0][-2:] # 示例：后2个批次用于测试
    # else:
    #     local_train_data, local_test_data = [], []
    #
    # # 示例：获取本地测试数据的非归一化标签（来自notebook）
    # label_test_data = [
    #    _get_data_for_name(building_data, building_names, name) # 使用原始数据获取标签
    #    for name in local_client_names
    # ]
    # if label_test_data and label_test_data[0]:
    #      label_test_data = label_test_data[0][-2:] # 标签的最后2个批次
    # else:
    #      label_test_data = []
    # --- 注释掉的本地数据处理结束 ---

    logger.info(f"为{len(federated_train_data)}个客户端准备联邦训练数据。")
    logger.info(f"为{len(federated_valid_data)}个客户端准备联邦验证数据。")

    # 返回数据集和计算的统计数据
    return federated_train_data, federated_valid_data, ec_min, ec_max, ec_mean #, local_data（如果取消上面的注释则添加）


if __name__ == '__main__':
    # 如果直接运行此脚本的示例用法
    logger.info("直接运行data_utils.py进行测试...")
    try:
        # 注意：config值EC_MIN, EC_MAX, EC_MEAN最初将为None
        # 函数preprocess_and_load_federated_data计算它们。
        train_data, valid_data, min_val, max_val, mean_val = preprocess_and_load_federated_data()

        # 更新配置值（如果需要，尽管通常由调用脚本完成）
        config.EC_MIN = min_val
        config.EC_MAX = max_val
        config.EC_MEAN = mean_val
        logger.info(f"更新配置统计数据: Min={config.EC_MIN:.4f}, Max={config.EC_MAX:.4f}, Mean={config.EC_MEAN:.4f}")


        print(f"\n训练客户端数量: {len(train_data)}")
        if train_data:
            print(f"第一个训练客户端的批次数量: {len(train_data[0])}")
            if train_data[0]:
                print(f"第一个训练客户端第一个批次的特征形状: {train_data[0][0]['x'].shape}")
                print(f"第一个训练客户端第一个批次的标签形状: {train_data[0][0]['y'].shape}")

        print(f"\n验证客户端数量: {len(valid_data)}")
        if valid_data:
            print(f"第一个验证客户端的批次数量: {len(valid_data[0])}")
            if valid_data[0]:
                 print(f"第一个验证客户端第一个批次的特征形状: {valid_data[0][0]['x'].shape}")
                 print(f"第一个验证客户端第一个批次的标签形状: {valid_data[0][0]['y'].shape}")

    except Exception as e:
        logger.error(f"直接执行期间发生错误: {e}", exc_info=True)
