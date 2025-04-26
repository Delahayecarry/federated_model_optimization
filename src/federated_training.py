import tensorflow as tf
import tensorflow_federated as tff
import logging
import os
import collections # 需要用于类型提示（如果使用）

# 使用相对导入导入src包内的模块
from . import config
from . import model_definition
from . import data_utils # 用于测试块需要

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def _ensure_dir_exists(path):
    """确保给定路径的目录存在。"""
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        logger.info(f"创建目录: {dir_name}")
        os.makedirs(dir_name, exist_ok=True)

def run_federated_training(federated_train_data: list, federated_valid_data: list):
    """
    构建联邦平均过程，运行训练循环，
    评估模型，实现早停，并保存最终模型。

    参数:
        federated_train_data: 用于训练的客户端数据集列表。
        federated_valid_data: 用于验证的客户端数据集列表。

    返回:
        训练后的最终服务器状态。如果训练设置失败，则返回None。
    """
    logger.info("--- 开始联邦训练过程 ---")

    # 1. 构建联邦平均过程
    # 使用model_definition中的model_fn和来自配置的优化器速率
    try:
        logger.info(f"构建联邦平均过程，客户端学习率={config.CLIENT_LEARNING_RATE}，服务器学习率={config.SERVER_LEARNING_RATE}")
        # 更新为tff0.74.0版本学习api
        '''
        # 新版API
iterative_process = tff.learning.algorithms.build_weighted_fed_avg(...)
evaluation = tff.learning.algorithms.build_fed_eval(...)
'''
        iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
            model_fn=model_definition.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=config.CLIENT_LEARNING_RATE),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=config.SERVER_LEARNING_RATE)
        )
        logger.info("联邦平均过程构建成功。")
        logger.debug(f"初始化类型签名: {iterative_process.initialize.type_signature}")
        logger.debug(f"下一步类型签名: {iterative_process.next.type_signature}")
    except Exception as e:
        logger.error(f"构建联邦平均过程失败: {e}", exc_info=True)
        return None

    # 2. 构建评估计算
    try:
        logger.info("构建联邦评估计算。")
        # 更新为tff0.74.0版本学习api
        '''
        # 新版API
        evaluation = tff.learning.algorithms.build_fed_eval(...)
        '''
        evaluation = tff.learning.algorithms.build_fed_eval(
            model_fn=model_definition.model_fn,
            metrics=tff.learning.metrics.sum_over_batch_size(tf.keras.metrics.MeanAbsoluteError())
        )
        logger.info("联邦评估计算构建成功。")
        logger.debug(f"评估类型签名: {evaluation.type_signature}")
    except Exception as e:
        logger.error(f"构建联邦评估计算失败: {e}", exc_info=True)
        return None

    # 3. 初始化服务器状态
    try:
        logger.info("初始化服务器状态。")
        state = iterative_process.initialize()
        logger.info("服务器状态初始化成功。")
    except Exception as e:
        logger.error(f"初始化服务器状态失败: {e}", exc_info=True)
        return None

    # 4. 运行训练循环
    logger.info(f"开始训练循环，共{config.NUM_ROUNDS}轮...")
    history = {'loss': [], 'mean_absolute_error': [], 'val_loss': [], 'val_mean_absolute_error': []}
    best_val_metric = float('inf')
    best_state = state
    rounds_without_improvement = 0
    early_stopping_patience = 5 # 定义早停的耐心值（可以移到配置中）

    for round_num in range(config.NUM_ROUNDS):
        try:
            # 运行一轮训练
            state, metrics = iterative_process.next(state, federated_train_data)
            train_metrics = metrics['train'] # TFF在'train'下嵌套训练指标
            history['loss'].append(float(train_metrics['loss']))
            history['mean_absolute_error'].append(float(train_metrics['mean_absolute_error']))

            # 在验证数据上评估
            valid_metrics = evaluation(state.model, federated_valid_data)
            val_loss = float(valid_metrics['loss'])
            val_mae = float(valid_metrics['mean_absolute_error']) # MAE是model_fn中定义的第二个指标
            history['val_loss'].append(val_loss)
            history['val_mean_absolute_error'].append(val_mae)

            logger.info(f"第{round_num+1:3d}/{config.NUM_ROUNDS}轮 - "
                        f"训练损失: {train_metrics['loss']:.4f}, 训练MAE: {train_metrics['mean_absolute_error']:.4f} | "
                        f"验证损失: {val_loss:.4f}, 验证MAE: {val_mae:.4f}")

            # 早停逻辑（基于验证MAE，就像notebook似乎隐式使用的那样，通过索引）
            # 使用MAE是因为越低越好，而且通常是主要的回归指标。
            # notebook代码使用valid_metrics[1]，这里对应于MAE。
            if val_mae < best_val_metric:
                improvement = best_val_metric - val_mae
                best_val_metric = val_mae
                best_state = state
                rounds_without_improvement = 0
                logger.info(f"  新的最佳验证MAE: {best_val_metric:.4f}。改进: {improvement:.4f}")
                # 如果需要，检查相对改进阈值（类似notebook）
                if len(history['val_mean_absolute_error']) > 1:
                    relative_improvement = (history['val_mean_absolute_error'][-2] - val_mae) / history['val_mean_absolute_error'][-2] if history['val_mean_absolute_error'][-2] != 0 else float('inf')
                    logger.info(f"  相对改进: {relative_improvement:.5f}")
                    if relative_improvement < config.EARLY_STOPPING_THRESHOLD:
                        logger.info(f"提前停止: 相对改进({relative_improvement:.5f})低于阈值({config.EARLY_STOPPING_THRESHOLD})。")
                        break

            else:
                rounds_without_improvement += 1
                logger.info(f"  验证MAE已连续{rounds_without_improvement}轮没有改进。最佳: {best_val_metric:.4f}")
                # 可选: 如果'patience'轮内没有改进则停止
                # if rounds_without_improvement >= early_stopping_patience:
                #     logger.info(f"提前停止: {early_stopping_patience}轮内没有改进。")
                #     break

        except Exception as e:
            logger.error(f"训练轮次{round_num + 1}期间出错: {e}", exc_info=True)
            # 决定是否要停止训练或尝试继续
            break # 出错时停止

    logger.info("--- 联邦训练循环完成 ---")
    logger.info(f"达到的最佳验证MAE: {best_val_metric:.4f}")

    # 5. 保存最终（最佳）模型
    try:
        logger.info("保存最佳模型权重...")
        # 创建一个Keras模型实例以分配权重
        local_fed_model = model_definition.create_keras_model()

        # 从最佳服务器状态分配权重
        '''
        # 旧版API
        tff.learning.assign_weights_to_keras_model(keras_model, state.model)

        # 新版API
        tff.learning.models.keras_utils.assign_weights_to_keras_model(keras_model, state.model)'''
        # 更新为tff0.74.0版本api
        tff.learning.models.keras_utils.assign_weights_to_keras_model(local_fed_model, best_state.model)
        logger.info("成功将权重分配给Keras模型。")

        # 从配置中定义保存路径
        # 使用os.path.join实现跨平台兼容性
        save_dir = config.MODEL_SAVE_BASE_DIR
        saved_model_path = os.path.join(save_dir, config.FEDERATED_MODEL_SAVE_NAME)
        h5_model_path = os.path.join(save_dir, config.FEDERATED_MODEL_H5_SAVE_NAME)

        # 确保目录存在
        _ensure_dir_exists(saved_model_path) # 检查SavedModel格式的目录
        _ensure_dir_exists(h5_model_path)    # 检查H5格式的目录

        # 以SavedModel格式保存
        logger.info(f"以SavedModel格式保存模型到: {saved_model_path}")
        tf.keras.models.save_model(local_fed_model, saved_model_path, save_format='tf') # 明确指定'tf'

        # 以H5格式保存
        logger.info(f"以H5格式保存模型到: {h5_model_path}")
        tf.keras.models.save_model(local_fed_model, h5_model_path, save_format='h5') # 明确指定'h5'

        logger.info("模型保存完成。")

    except Exception as e:
        logger.error(f"保存最终模型失败: {e}", exc_info=True)

    return best_state # 返回找到的最佳状态


if __name__ == '__main__':
    # 直接测试此模块的示例用法
    # 需要先加载数据。
    logger.info("--- 直接运行federated_training.py进行测试 ---")

    # 设置TFF执行器（如果独立运行需要）- 选择适当的执行器
    # tff.backends.native.set_local_python_execution_context() # 示例: 原生后端

    try:
        # 1. 使用data_utils加载数据
        # 注意: data_utils计算EC统计数据但这里不直接修改配置。
        train_data, valid_data, _, _, _ = data_utils.preprocess_and_load_federated_data()

        if not train_data or not valid_data:
             logger.error("加载训练或验证数据失败。退出测试。")
        else:
             # 2. 运行联邦训练
             final_state = run_federated_training(train_data, valid_data)

             if final_state:
                 logger.info("联邦训练测试成功完成。")
                 # logger.info(f"最终状态结构: {final_state}") # 可能非常冗长
             else:
                 logger.error("联邦训练测试失败。")

    except Exception as e:
        logger.error(f"直接执行测试期间发生错误: {e}", exc_info=True)
