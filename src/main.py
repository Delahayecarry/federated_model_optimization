#!/usr/bin/env python
# fedfederated_model_optimization/src/main.py
"""
联邦学习项目的主入口点。
"""
import argparse
import logging
import os
import tensorflow_federated as tff
import tensorflow as tf
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),
        logging.FileHandler(filename='federated_learning.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)
 
# 导入项目模块
from src import config
from src import data_utils
from src import federated_training
from src import evaluation

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description='联邦学习训练与评估')
    
    # 定义主要模式（训练或评估）
    parser.add_argument('mode', choices=['train', 'evaluate', 'hypertune'], 
                        help='运行模式: train=训练新模型, evaluate=评估已有模型, hypertune=超参数优化')
    
    # 数据相关参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--data_path', type=str, help='数据文件路径')
    data_group.add_argument('--batch_size', type=int, help='批处理大小')
    data_group.add_argument('--num_examples_per_user', type=int, help='每个客户端的最大样本数')
    
    # 模型相关参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--num_layers', type=int, help='隐藏层数量')
    model_group.add_argument('--num_neurons', type=int, help='每层神经元数量')
    model_group.add_argument('--activation', type=str, 
                            choices=['relu', 'sigmoid', 'tanh'], 
                            help='激活函数类型')
    
    # 训练相关参数
    train_group = parser.add_argument_group('训练参数')
    train_group.add_argument('--num_rounds', type=int, help='联邦学习轮数')
    train_group.add_argument('--client_lr', type=float, help='客户端学习率')
    train_group.add_argument('--server_lr', type=float, help='服务器学习率')
    train_group.add_argument('--early_stopping_threshold', type=float, 
                             help='早停阈值（相对改进）')
    
    # 保存/加载相关参数
    io_group = parser.add_argument_group('模型存储参数')
    io_group.add_argument('--model_save_dir', type=str, help='模型保存基础目录')
    io_group.add_argument('--model_name', type=str, help='模型名称')
    io_group.add_argument('--model_path', type=str, help='要评估的模型路径 (仅评估模式)')
    
    # 超参数优化相关参数
    hyper_group = parser.add_argument_group('超参数优化参数 (hypertune模式)')
    hyper_group.add_argument('--search_layers', type=str, help='要尝试的层数,逗号分隔 (例如: "1,2,3")')
    hyper_group.add_argument('--search_neurons', type=str, help='要尝试的神经元数,逗号分隔 (例如: "5,10,15")')
    hyper_group.add_argument('--search_activations', type=str, 
                             help='要尝试的激活函数,逗号分隔 (例如: "relu,tanh,sigmoid")')
    
    return parser.parse_args()

def update_config_from_args(args):
    """根据命令行参数更新配置模块的值。"""
    # 仅在提供参数（不为None）时更新配置值
    if args.data_path is not None:
        config.DATA_PATH = args.data_path
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.num_examples_per_user is not None:
        config.NUM_EXAMPLES_PER_USER = args.num_examples_per_user
    
    if args.num_layers is not None:
        config.NUM_LAYERS = args.num_layers
    if args.num_neurons is not None:
        config.NUM_NEURONS = args.num_neurons
    if args.activation is not None:
        config.ACTIVATION = args.activation
    
    if args.num_rounds is not None:
        config.NUM_ROUNDS = args.num_rounds
    if args.client_lr is not None:
        config.CLIENT_LEARNING_RATE = args.client_lr
    if args.server_lr is not None:
        config.SERVER_LEARNING_RATE = args.server_lr
    if args.early_stopping_threshold is not None:
        config.EARLY_STOPPING_THRESHOLD = args.early_stopping_threshold
    
    if args.model_save_dir is not None:
        config.MODEL_SAVE_BASE_DIR = args.model_save_dir
    if args.model_name is not None:
        # 根据提供的model_name更新SavedModel和H5名称
        config.FEDERATED_MODEL_SAVE_NAME = args.model_name
        config.FEDERATED_MODEL_H5_SAVE_NAME = f"{args.model_name}.h5"
    
    # 记录将要使用的配置
    logger.info("----- 配置参数 -----")
    logger.info(f"数据路径: {config.DATA_PATH}")
    logger.info(f"批大小: {config.BATCH_SIZE}")
    logger.info(f"每客户端样本数: {config.NUM_EXAMPLES_PER_USER}")
    logger.info(f"模型结构: {config.NUM_LAYERS}层, 每层{config.NUM_NEURONS}个神经元, '{config.ACTIVATION}'激活函数")
    logger.info(f"训练参数: {config.NUM_ROUNDS}轮, 客户端LR={config.CLIENT_LEARNING_RATE}, 服务器LR={config.SERVER_LEARNING_RATE}")
    if args.mode == 'train' or args.mode == 'hypertune':
        logger.info(f"模型保存位置: {config.MODEL_SAVE_BASE_DIR}/{config.FEDERATED_MODEL_SAVE_NAME}")
    logger.info("---------------------")

def run_training():
    """运行训练过程。"""
    logger.info("开始加载数据...")
    # 加载数据
    try:
        train_data, valid_data, ec_min, ec_max, ec_mean = data_utils.preprocess_and_load_federated_data()
        # 使用计算得到的EC值更新配置
        config.EC_MIN = ec_min
        config.EC_MAX = ec_max
        config.EC_MEAN = ec_mean
        logger.info(f"数据加载完成，数据统计信息: MIN={ec_min:.4f}, MAX={ec_max:.4f}, MEAN={ec_mean:.4f}")
    except Exception as e:
        logger.error(f"数据加载失败: {e}", exc_info=True)
        return False
    
    # 设置TFF执行器 - 匹配notebook的执行器设置
    try:
        logger.info("设置TensorFlow Federated执行环境...")
        # 更新为tff0.74.0版本
        tff.backends.native.create_sync_local_cpp_execution_context()
    except Exception as e:
        logger.error(f"TFF执行环境设置失败: {e}", exc_info=True)
        return False
    
    # 运行联邦训练
    logger.info("开始联邦训练...")
    try:
        final_state = federated_training.run_federated_training(train_data, valid_data)
        if final_state:
            logger.info("联邦训练完成！")
            return True
        else:
            logger.error("联邦训练失败，未返回有效状态。")
            return False
    except Exception as e:
        logger.error(f"联邦训练过程中出现错误: {e}", exc_info=True)
        return False

def run_hyperparameter_tuning(args):
    """运行超参数优化过程。"""
    logger.info("开始超参数优化...")
    
    # 解析超参数搜索空间
    layers_to_try = [int(x) for x in args.search_layers.split(',')] if args.search_layers else [1, 3, 5]
    neurons_to_try = [int(x) for x in args.search_neurons.split(',')] if args.search_neurons else [5, 10, 15, 20, 25]
    activations_to_try = args.search_activations.split(',') if args.search_activations else ['relu', 'tanh', 'sigmoid']
    
    logger.info("超参数搜索空间: ")
    logger.info(f"  - 层数: {layers_to_try}")
    logger.info(f"  - 神经元数: {neurons_to_try}")
    logger.info(f"  - 激活函数: {activations_to_try}")
    
    # 为所有实验一次性加载数据
    try:
        logger.info("加载数据...")
        train_data, valid_data, ec_min, ec_max, ec_mean = data_utils.preprocess_and_load_federated_data()
        # 使用计算得到的EC值更新配置
        config.EC_MIN = ec_min
        config.EC_MAX = ec_max
        config.EC_MEAN = ec_mean
    except Exception as e:
        logger.error(f"数据加载失败: {e}", exc_info=True)
        return False
    
    # 设置TFF执行器
    try:
        logger.info("设置TensorFlow Federated执行环境...")
        local_context = tff.backends.native.create_local_execution_context()

        # 设置默认执行上下文
        tff.framework.set_default_context(local_context)
    except Exception as e:
        logger.error(f"TFF执行环境设置失败: {e}", exc_info=True)
        return False
    
    # 跟踪最佳模型
    best_metric = float('inf')
    best_config = None
    
    # 循环遍历超参数组合
    for num_layers in layers_to_try:
        for num_neurons in neurons_to_try:
            for activation in activations_to_try:
                logger.info(f"尝试参数组合: 层数={num_layers}, 神经元数={num_neurons}, 激活函数={activation}")
                
                # 为此次运行更新配置
                config.NUM_LAYERS = num_layers
                config.NUM_NEURONS = num_neurons
                config.ACTIVATION = activation
                
                # 根据超参数创建模型保存名称
                config.FEDERATED_MODEL_SAVE_NAME = f"model_L{num_layers}_N{num_neurons}_{activation}"
                config.FEDERATED_MODEL_H5_SAVE_NAME = f"{config.FEDERATED_MODEL_SAVE_NAME}.h5"
                
                try:
                    # 使用当前超参数运行训练
                    final_state = federated_training.run_federated_training(train_data, valid_data)
                    
                    if final_state:
                        # 评估模型
                        # 我们需要在这里加载并评估保存的模型
                        # 假设训练过程中的验证结果确定最佳模型
                        current_metric = federated_training.get_best_metric(final_state)
                        logger.info(f"参数组合表现 - 验证指标: {current_metric:.4f}")
                        
                        if current_metric < best_metric:
                            best_metric = current_metric
                            best_config = {
                                'num_layers': num_layers,
                                'num_neurons': num_neurons,
                                'activation': activation,
                                'metric': current_metric
                            }
                            logger.info(f"发现新的最佳参数组合! 验证指标: {best_metric:.4f}")
                    
                except Exception as e:
                    logger.error(f"参数组合训练失败: {e}", exc_info=True)
                    continue
                
                # 强制TF清理一些内存
                tf.keras.backend.clear_session()
    
    # 记录最终结果
    if best_config:
        logger.info("===== 超参数优化完成 =====")
        logger.info(f"最佳参数组合: 层数={best_config['num_layers']}, " 
                    f"神经元数={best_config['num_neurons']}, "
                    f"激活函数={best_config['activation']}")
        logger.info(f"最佳验证指标: {best_config['metric']:.4f}")
        
        # 将最佳配置保存到文件
        try:
            with open('best_hyperparameters.txt', 'w') as f:
                f.write(f"层数: {best_config['num_layers']}\n")
                f.write(f"神经元数: {best_config['num_neurons']}\n")
                f.write(f"激活函数: {best_config['activation']}\n")
                f.write(f"验证指标: {best_config['metric']:.4f}\n")
            logger.info("最佳参数已保存到 'best_hyperparameters.txt'")
        except Exception as e:
            logger.error(f"保存最佳参数失败: {e}")
        
        return True
    else:
        logger.error("超参数优化未产生有效结果")
        return False

def run_evaluation(args):
    """Run the evaluation process."""
    # Determine which model to evaluate
    model_path = args.model_path
    
    if model_path is None:
        # Use the default path from config
        model_path = os.path.join(config.MODEL_SAVE_BASE_DIR, config.FEDERATED_MODEL_H5_SAVE_NAME)
    
    logger.info(f"将评估模型: {model_path}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    # Load test data
    try:
        logger.info("加载测试数据...")
        # Modify this call based on your actual data_utils implementation
        # Here we assume a function that returns a test set specifically
        train_data, valid_data, ec_min, ec_max, ec_mean, test_x, test_y = data_utils.preprocess_and_load_federated_data(include_test=True)
        logger.info(f"测试数据加载完成: 形状 X={test_x.shape}, Y={test_y.shape}")
    except Exception as e:
        logger.error(f"测试数据加载失败: {e}", exc_info=True)
        return False
    
    # Evaluate model
    try:
        logger.info("开始模型评估...")
        metrics = evaluation.evaluate_model(
            model_path=model_path,
            test_data_x=test_x,
            test_data_y_normalized=test_y,
            ec_min=ec_min,
            ec_max=ec_max
        )
        
        if metrics:
            logger.info("===== 评估结果 =====")
            logger.info(f"MSE (标准化): {metrics['loss']:.6f}")
            logger.info(f"MAE (标准化): {metrics['mae']:.6f}")
            logger.info(f"CVRMSE (非标准化): {metrics['cvrmse']:.6f}")
            return True
        else:
            logger.error("评估失败，未返回有效指标。")
            return False
    except Exception as e:
        logger.error(f"评估过程中出现错误: {e}", exc_info=True)
        return False

def main():
    """Main entry point function."""
    args = parse_args()
    
    # Update config with command line arguments
    update_config_from_args(args)
    
    # Run the requested mode
    if args.mode == 'train':
        logger.info("=== 开始模型训练 ===")
        success = run_training()
    elif args.mode == 'evaluate':
        logger.info("=== 开始模型评估 ===")
        success = run_evaluation(args)
    elif args.mode == 'hypertune':
        logger.info("=== 开始超参数优化 ===")
        success = run_hyperparameter_tuning(args)
    else:
        logger.error(f"未知模式: {args.mode}")
        return 1
    
    if success:
        logger.info(f"=== {args.mode} 模式运行成功 ===")
        return 0
    else:
        logger.error(f"=== {args.mode} 模式运行失败 ===")
        return 1

# This allows the script to be run directly or imported as a module
if __name__ == "__main__":
    # When running directly, main() return value becomes exit code
    sys.exit(main())
