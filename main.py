import tensorflow as tf
import tensorflow_federated as tff
import logging
import os
import sys
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=全部日志，1=不显示INFO，2=不显示INFO和WARNING，3=不显示所有

# 将src添加到路径中
sys.path.append(os.path.join(os.path.dirname(__file__)))

# 导入项目模块
from src import data_utils
from src import federated_training
from src import model_definition
from src import config

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """程序入口点"""
    # 显示版本信息但隐藏其他TF日志
    logger.info(f"TensorFlow版本: {tf.__version__}")
    logger.info(f"TensorFlow Federated版本: {tff.__version__}")
    
    # 设置日志级别，抑制警告
    logging.getLogger().setLevel(logging.ERROR)  # 只显示错误
    tf.get_logger().setLevel(logging.ERROR)  # 减少TensorFlow的日志
    
    # 只允许我们的主logger显示INFO
    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.getLogger('src.federated_training').setLevel(logging.INFO)
    
    # 设置TFF本地执行环境
    logger.info("设置TFF本地执行环境")
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
    
    # 加载数据
    logger.info("加载联邦数据集")
    train_data, valid_data, _, _, _ = data_utils.preprocess_and_load_federated_data()
    
    if not train_data or not valid_data:
        logger.error("加载训练或验证数据失败")
        return
    
    # 运行联邦训练
    logger.info("========== 开始联邦训练过程 ==========")
    start_time = tf.timestamp()
    final_state = federated_training.run_federated_training(train_data, valid_data)
    end_time = tf.timestamp()
    training_time = end_time - start_time
    
    if final_state:
        logger.info("联邦训练成功完成")
        logger.info(f"训练总耗时: {training_time:.2f} 秒")
        
        # 获取训练历史并显示
        if hasattr(final_state, 'history'):
            history = final_state.history
            logger.info("训练历史:")
            for key, values in history.items():
                if values:  # 确保列表不为空
                    logger.info(f"  {key}: 开始={values[0]:.4f}, 结束={values[-1]:.4f}, 最佳={min(values):.4f if key.startswith('loss') or key.startswith('val_loss') else max(values):.4f}")
        
        # 显示模型结构
        logger.info("模型权重结构:")
        for i, weight in enumerate(final_state.global_model_weights.trainable):
            logger.info(f"  层 {i}: 形状={weight.shape}")
            
        # 检查保存的模型
        logger.info("检查保存的模型文件:")
        saved_model_path = os.path.join(config.MODEL_SAVE_BASE_DIR, config.FEDERATED_MODEL_SAVE_NAME)
        h5_model_path = os.path.join(config.MODEL_SAVE_BASE_DIR, config.FEDERATED_MODEL_H5_SAVE_NAME)
        
        if os.path.exists(saved_model_path):
            logger.info(f"  SavedModel模型已保存到: {saved_model_path}")
        else:
            logger.warning(f"  SavedModel模型未找到: {saved_model_path}")
            
        if os.path.exists(h5_model_path):
            logger.info(f"  H5模型已保存到: {h5_model_path}, 大小: {os.path.getsize(h5_model_path)/1024:.1f} KB")
        else:
            logger.warning(f"  H5模型未找到: {h5_model_path}")
    else:
        logger.error("联邦训练失败")

if __name__ == "__main__":
    main()
