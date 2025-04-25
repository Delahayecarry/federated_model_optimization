import tensorflow as tf
import numpy as np
import logging
import os

# 使用相对导入导入src包内的模块
from . import config
# 如果需要专门在此处加载测试数据，则导入data_utils
# from . import data_utils
# 除非重新创建模型结构，否则可能不需要严格导入model_definition
# from . import model_definition

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def calculate_cvrmse(predictions: np.ndarray, true_values: np.ndarray) -> float:
    """
    计算均方根误差变异系数（CVRMSE）。

    参数:
        predictions: 预测值的numpy数组。
        true_values: 真实值的numpy数组。

    返回:
        CVRMSE分数，以浮点数表示。如果true_values的平均值为零，则返回NaN。
    """
    if predictions.shape != true_values.shape:
        raise ValueError(f"预测形状{predictions.shape}必须与真实值形状{true_values.shape}匹配")
    if len(predictions) == 0:
        return np.nan # 或抛出错误？

    rmse = np.sqrt(np.mean(np.square(true_values - predictions)))
    mean_true = np.mean(true_values)

    if mean_true == 0:
        logger.warning("真实值的平均值为零，CVRMSE未定义（返回NaN）。")
        return np.nan
    else:
        cvrmse = rmse / mean_true
        return float(cvrmse)

def evaluate_model(model_path: str,
                   test_data_x: np.ndarray,
                   test_data_y_normalized: np.ndarray,
                   ec_min: float,
                   ec_max: float):
    """
    加载保存的Keras模型，在归一化的测试数据上评估它，
    在反归一化的预测上计算CVRMSE，并记录结果。

    参数:
        model_path: 保存的Keras模型路径（H5或SavedModel目录）。
        test_data_x: 归一化测试特征的numpy数组。
        test_data_y_normalized: 归一化测试标签的numpy数组。
        ec_min: 用于目标变量归一化的最小值。
        ec_max: 用于目标变量归一化的最大值。

    返回:
        包含评估指标的字典（'mse'、'mae'、'cvrmse'）。
        如果评估失败，则返回None。
    """
    logger.info(f"--- 开始模型评估: {model_path} ---")

    if not os.path.exists(model_path):
        logger.error(f"未找到模型路径: {model_path}")
        return None
    if ec_min is None or ec_max is None:
        logger.error("必须提供EC_MIN和EC_MAX用于反归一化。")
        return None
    if ec_max == ec_min:
        logger.error("EC_MAX和EC_MIN不能相等，用于反归一化。")
        return None

    try:
        # 1. 加载保存的Keras模型
        logger.info(f"从{model_path}加载模型...")
        model = tf.keras.models.load_model(model_path)
        logger.info("模型加载成功。")
        model.summary(print_fn=logger.info)

        # 2. 在*归一化*数据上评估（标准Keras指标）
        logger.info("在归一化测试数据上评估模型（MSE, MAE）...")
        # 假设模型已使用'mse'和'mae'指标编译
        try:
            eval_results = model.evaluate(test_data_x, test_data_y_normalized, verbose=0)
            # 确保根据模型编译正确命名指标
            # 默认索引：0=损失（mse），1=mae（通常）
            metrics = {'loss': eval_results[0], 'mae': eval_results[1]} # 如有需要调整索引
            logger.info(f"归一化评估 - 损失（MSE）: {metrics['loss']:.4f}, MAE: {metrics['mae']:.4f}")
        except Exception as e:
             logger.warning(f"无法运行model.evaluate（模型可能需要重新编译或指标不同）: {e}")
             metrics = {'loss': np.nan, 'mae': np.nan}


        # 3. 进行预测（归一化）
        logger.info("在测试数据上生成预测...")
        predictions_normalized = model.predict(test_data_x)
        # 如有必要确保预测被展平（例如，形状（N, 1）->（N,））
        predictions_normalized = predictions_normalized.flatten()
        logger.info(f"生成了{len(predictions_normalized)}个预测。")

        # 4. 反归一化预测和真实标签
        logger.info("反归一化预测和真实标签...")
        predictions_actual = predictions_normalized * (ec_max - ec_min) + ec_min
        true_actual = test_data_y_normalized * (ec_max - ec_min) + ec_min

        # 5. 在反归一化数据上计算CVRMSE
        logger.info("计算CVRMSE...")
        cvrmse = calculate_cvrmse(predictions_actual, true_actual)
        metrics['cvrmse'] = cvrmse
        logger.info(f"CVRMSE（在反归一化数据上）: {cvrmse:.4f}")

        logger.info("--- 模型评估完成 ---")
        return metrics

    except Exception as e:
        logger.error(f"评估期间发生错误: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    # 直接测试此模块的示例用法
    # 这需要修改data_utils以实际生成测试拆分
    # 或在此手动加载数据。
    logger.info("--- 直接运行evaluate.py进行测试 ---")

    # --- 测试配置 ---
    # !!! 此部分需要根据您如何管理测试数据进行调整 !!!

    # 选项1：假设data_utils.py已被修改以返回测试数据
    # try:
    #     from . import data_utils # 确保data_utils可导入
    #     # 修改preprocess_and_load_federated_data以返回测试数据拆分
    #     # 例如，_, _, ec_min, ec_max, _, test_x, test_y_norm = data_utils.preprocess_and_load_federated_data(include_test=True)
    #     logger.warning("直接执行需要修改data_utils.py以返回测试数据。")
    #     # 占位符 - 替换为实际数据加载
    #     test_x = np.random.rand(100, config.INPUT_SHAPE[0]).astype(np.float32)
    #     test_y_norm = np.random.rand(100).astype(np.float32)
    #     ec_min_test = 0 # 占位符
    #     ec_max_test = 1 # 占位符
    # except ImportError:
    #      logger.error("无法导入data_utils。确保它在src目录中且__init__.py存在。")
    #      exit(1)
    # except Exception as e:
    #      logger.error(f"加载测试数据时出错: {e}")
    #      exit(1)

    # 选项2：手动定义路径并加载数据（不太理想）
    # 定义要评估的保存模型的路径（替换为您实际的保存模型路径）
    model_to_evaluate = os.path.join(config.MODEL_SAVE_BASE_DIR, config.FEDERATED_MODEL_H5_SAVE_NAME) # 示例：使用H5模型

    # 手动加载测试数据和EC值（替换为您的实际逻辑）
    logger.warning("直接执行测试使用占位符测试数据和EC值。")
    try:
        # 通常情况下，您应在此处加载您的*实际*测试数据和EC常量
        # 为了演示，我们创建随机数据和虚拟EC值
        test_x = np.random.rand(200, config.INPUT_SHAPE[0]).astype(np.float32) # 示例测试特征
        # 基于某些线性关系+噪声模拟归一化标签
        true_weights = np.random.rand(config.INPUT_SHAPE[0]).astype(np.float32)
        true_bias = np.random.rand(1).astype(np.float32)
        test_y_actual_example = np.dot(test_x, true_weights) + true_bias + np.random.normal(0, 0.1, 200).astype(np.float32)
        # 在此示例中为归一化/反归一化定义虚拟EC最小/最大值
        ec_min_test = float(np.min(test_y_actual_example)) - 1.0 # 添加缓冲
        ec_max_test = float(np.max(test_y_actual_example)) + 1.0 # 添加缓冲
        # 归一化示例标签
        test_y_norm = (test_y_actual_example - ec_min_test) / (ec_max_test - ec_min_test)

        logger.info(f"使用虚拟测试数据: X形状{test_x.shape}, Y_norm形状{test_y_norm.shape}")
        logger.info(f"使用虚拟EC值: Min={ec_min_test:.4f}, Max={ec_max_test:.4f}")

    except Exception as e:
        logger.error(f"准备手动测试数据时出错: {e}", exc_info=True)
        exit(1)
    # --- 配置结束 ---


    # 在尝试评估前检查模型文件是否存在
    if not os.path.exists(model_to_evaluate):
         logger.error(f"测试用的模型文件未找到: {model_to_evaluate}")
         logger.error("请确保已运行联邦训练并保存了模型，或更新路径。")
    else:
        # 运行评估
        evaluation_metrics = evaluate_model(
            model_path=model_to_evaluate,
            test_data_x=test_x,
            test_data_y_normalized=test_y_norm,
            ec_min=ec_min_test,
            ec_max=ec_max_test
        )

        if evaluation_metrics:
            logger.info(f"评估测试完成。指标: {evaluation_metrics}")
        else:
            logger.error("评估测试失败。")
