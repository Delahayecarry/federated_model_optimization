import tensorflow as tf
import tensorflow_federated as tff
import collections
import logging

# 使用相对导入访问同一包中的配置
from . import config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义模型使用的数据批次规格。
# 这需要与data_utils.py生成的字典结构匹配
BATCH_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(shape=[None, config.INPUT_SHAPE[0]], dtype=tf.float32),
    y=tf.TensorSpec(shape=[None], dtype=tf.float32)
)
logger.info(f"定义BATCH_SPEC，输入形状: {BATCH_SPEC['x'].shape}")

def create_keras_model():
    """
    基于配置模块中的参数创建一个顺序Keras模型。

    返回:
        一个tf.keras.models.Sequential模型。
    """
    logger.info(f"创建Keras模型，包含{config.NUM_LAYERS}个隐藏层，"
                f"每层{config.NUM_NEURONS}个神经元，"
                f"'{config.ACTIVATION}'激活函数。")

    model = tf.keras.models.Sequential()
    # 输入层由第一个Dense层的input_shape隐式定义
    # 第一个隐藏层
    model.add(tf.keras.layers.Dense(
        config.NUM_NEURONS,
        activation=config.ACTIVATION,
        input_shape=config.INPUT_SHAPE # 为第一层定义输入形状
    ))
    # 额外的隐藏层
    for _ in range(config.NUM_LAYERS - 1):
        model.add(tf.keras.layers.Dense(
            config.NUM_NEURONS,
            activation=config.ACTIVATION
        ))
    # 输出层（回归任务的单个神经元，线性激活）
    model.add(tf.keras.layers.Dense(1))

    logger.info("Keras模型创建成功。")
    model.summary(print_fn=logger.info) # 记录模型摘要
    return model

def model_fn():
    """
    TFF模型函数。创建一个Keras模型并将其包装为tff.learning.Model。

    该函数将被TFF在不同的图上下文中调用。
    *不要*从外部作用域捕获预先创建的Keras模型，这一点至关重要。

    返回:
        一个tff.learning.Model。
    """
    # 在此函数内创建一个新的Keras模型实例。
    keras_model = create_keras_model()

    # 将Keras模型包装以在TFF中使用。
    # 在这里定义损失和度量。
    return tff.learning.from_keras_model(
        keras_model=keras_model,
        input_spec=BATCH_SPEC,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()] # 与notebook中的指标匹配
    )

if __name__ == '__main__':
    # 如果直接运行此脚本进行测试的示例用法
    logger.info("直接运行model_definition.py进行测试...")

    # 测试Keras模型创建
    keras_m = create_keras_model()
    print("\nKeras模型摘要:")
    keras_m.summary()

    # 测试TFF模型函数
    tff_model = model_fn()
    print(f"\nTFF模型类型签名: {tff_model.type_signature}")
    # 注意：您不能在这里轻易直接"运行"或"预测"tff_model对象。
    # 它被设计用于TFF计算中，如build_federated_averaging_process。
    logger.info("TFF模型函数执行成功。")
