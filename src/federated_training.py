import tensorflow as tf
import tensorflow_federated as tff
import logging
import os
import warnings

# 抑制警告和TF日志
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)

# 使用相对导入导入src包内的模块
from . import config
from . import model_definition
from . import data_utils # 用于测试块需要


# 配置日志 - 使用特定于此模块的logger
logger = logging.getLogger(__name__)

# 提供一个函数输出漂亮的进度条，减少需要的日志信息
def print_progress(current, total, message="", bar_length=30):
    """在控制台打印进度条"""
    percent = float(current) / total
    arrow = "-" * int(round(percent * bar_length))
    spaces = " " * (bar_length - len(arrow))
    
    # 使用\r覆盖同一行，避免日志过多
    print(f"\r[{arrow}{spaces}] {int(round(percent * 100))}% {message}", end="", flush=True)
    if current == total:
        print()  # 添加换行符

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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 在此块中抑制所有警告
            iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
                model_fn=model_definition.model_fn,
                client_optimizer_fn=tff.learning.optimizers.build_sgdm(
                    learning_rate=config.CLIENT_LEARNING_RATE
                ),
                server_optimizer_fn=tff.learning.optimizers.build_sgdm(
                    learning_rate=config.SERVER_LEARNING_RATE
                )
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
        )
        logger.info("联邦评估计算构建成功。")
        logger.debug(f"评估类型签名: {evaluation.initialize.type_signature}")
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
    evaluation_state = evaluation.initialize()  # 初始化评估状态
    best_val_metric = float('inf')
    best_state = state
    rounds_without_improvement = 0
    early_stopping_patience = 5 # 定义早停的耐心值（可以移到配置中）

    for round_num in range(config.NUM_ROUNDS):
        try:
            # 显示进度条
            print_progress(round_num, config.NUM_ROUNDS, f"训练中... 轮次 {round_num+1}/{config.NUM_ROUNDS}")
            
            # 运行一轮训练
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略训练内部的警告
                state, metrics = iterative_process.next(state, federated_train_data)

            # 使用正确的嵌套结构访问训练指标
            train_metrics = metrics['client_work']['train']
            history['loss'].append(float(train_metrics['loss']))
            history['mean_absolute_error'].append(float(train_metrics['mean_absolute_error']))

            # 在验证数据上评估 - 使用新的评估API
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # 忽略评估内部的警告
                evaluation_result = evaluation.next(evaluation_state, federated_valid_data)
            
            evaluation_state = evaluation_result.state
            valid_metrics = evaluation_result.metrics
            
            # 输出完整的验证指标结构用于调试
            logger.info(f"验证指标结构: {valid_metrics}")
            
            # 更新指标访问路径以匹配新的API结构
            try:
                # 先添加一个递归搜索函数，帮助找到指标
                def find_metrics_in_dict(metrics_dict, target_keys, path=""):
                    """
                    递归搜索字典中的指标键
                    返回找到的第一个目标键的值和路径
                    """
                    results = {}
                    
                    if not isinstance(metrics_dict, dict):
                        return results
                    
                    # 先检查当前级别
                    for key in target_keys:
                        if key in metrics_dict:
                            results[key] = {
                                'value': metrics_dict[key],
                                'path': f"{path}.{key}" if path else key
                            }
                    
                    # 如果没有找到所有目标键，递归搜索下一级
                    if len(results) < len(target_keys):
                        for k, v in metrics_dict.items():
                            if isinstance(v, dict):
                                new_path = f"{path}.{k}" if path else k
                                sub_results = find_metrics_in_dict(v, target_keys, new_path)
                                # 合并结果，但不覆盖已找到的键
                                for sub_key, sub_value in sub_results.items():
                                    if sub_key not in results:
                                        results[sub_key] = sub_value
                    
                    return results
                
                # 搜索loss和mean_absolute_error键
                target_metrics = ['loss', 'mean_absolute_error']
                found_metrics = find_metrics_in_dict(valid_metrics, target_metrics)
                
                logger.debug(f"找到的指标: {found_metrics}")
                
                # 提取找到的指标值
                if 'loss' in found_metrics:
                    val_loss = float(found_metrics['loss']['value'])
                    logger.info(f"从路径 {found_metrics['loss']['path']} 获取损失值")
                else:
                    # 如果找不到损失，尝试替代指标或使用NaN
                    logger.warning("找不到loss指标，尝试其他可能的指标名称")
                    
                    # 尝试其他可能的损失指标名称
                    mse_metrics = find_metrics_in_dict(valid_metrics, ['mean_squared_error', 'mse'])
                    if 'mean_squared_error' in mse_metrics:
                        val_loss = float(mse_metrics['mean_squared_error']['value'])
                        logger.info(f"使用 {mse_metrics['mean_squared_error']['path']} 作为替代损失指标")
                    elif 'mse' in mse_metrics:
                        val_loss = float(mse_metrics['mse']['value'])
                        logger.info(f"使用 {mse_metrics['mse']['path']} 作为替代损失指标")
                    else:
                        val_loss = float('nan')
                        logger.warning("无法找到任何损失相关指标，使用NaN")
                
                if 'mean_absolute_error' in found_metrics:
                    val_mae = float(found_metrics['mean_absolute_error']['value'])
                    logger.info(f"从路径 {found_metrics['mean_absolute_error']['path']} 获取MAE值")
                else:
                    # 如果找不到MAE，尝试替代指标或使用NaN
                    logger.warning("找不到mean_absolute_error指标，尝试其他可能的指标名称")
                    
                    # 尝试其他可能的MAE指标名称
                    mae_metrics = find_metrics_in_dict(valid_metrics, ['mae'])
                    if 'mae' in mae_metrics:
                        val_mae = float(mae_metrics['mae']['value'])
                        logger.info(f"使用 {mae_metrics['mae']['path']} 作为替代MAE指标")
                    else:
                        val_mae = float('nan')
                        logger.warning("无法找到任何MAE相关指标，使用NaN")
            except Exception as e:
                logger.error(f"访问验证指标时出错: {e}", exc_info=True)
                val_loss = float('nan')
                val_mae = float('nan')
                
            # 记录提取的指标到历史记录
            history['val_loss'].append(val_loss)
            history['val_mean_absolute_error'].append(val_mae)

            # 打印训练进度摘要 - 改用单行输出，避免过多日志
            progress_msg = (f"第{round_num+1:3d}/{config.NUM_ROUNDS}轮 - "
                            f"训练: loss={train_metrics['loss']:.4f}, MAE={train_metrics['mean_absolute_error']:.4f} | "
                            f"验证: loss={val_loss:.4f}, MAE={val_mae:.4f}")
            print(f"\r{progress_msg}", end="", flush=True)

            # 早停逻辑
            if val_mae < best_val_metric:
                improvement = best_val_metric - val_mae
                best_val_metric = val_mae
                best_state = state
                rounds_without_improvement = 0
                # 对于改进，换一行打印，使其更明显
                print(f"\n  新的最佳验证MAE: {best_val_metric:.4f}。改进: {improvement:.4f}")
                # 如果需要，检查相对改进阈值（类似notebook）
                if len(history['val_mean_absolute_error']) > 1:
                    relative_improvement = (history['val_mean_absolute_error'][-2] - val_mae) / history['val_mean_absolute_error'][-2] if history['val_mean_absolute_error'][-2] != 0 else float('inf')
                    print(f"  相对改进: {relative_improvement:.5f}")
                    if relative_improvement < config.EARLY_STOPPING_THRESHOLD:
                        print(f"\n提前停止: 相对改进({relative_improvement:.5f})低于阈值({config.EARLY_STOPPING_THRESHOLD})。")
                        break

            else:
                rounds_without_improvement += 1
                print(f"\n  验证MAE已连续{rounds_without_improvement}轮没有改进。最佳: {best_val_metric:.4f}")
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
        local_fed_model.set_weights(best_state.global_model_weights.trainable)
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
        # 尝试新的API路径
        if hasattr(tff.backends.native, 'set_sync_local_cpp_execution_context'):
            logger.info("使用set_sync_local_cpp_execution_context")
            tff.backends.native.set_sync_local_cpp_execution_context()
        elif hasattr(tff.backends.native, 'set_local_execution_context'):
            logger.info("使用set_local_execution_context")
            tff.backends.native.set_local_execution_context()
        else:
            # 尝试其他可能的API
            logger.warning("找不到标准TFF执行上下文API，尝试其他方法")
            if hasattr(tff.backends, 'native'):
                logger.info(f"可用的native后端方法: {dir(tff.backends.native)}")
                # 使用最可能的替代方法
                tff.backends.native.set_sync_local_cpp_execution_context()
    except Exception as e:
        logger.error(f"设置TFF执行上下文时出错: {e}")
        logger.error("继续运行，但可能会遇到问题")

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
