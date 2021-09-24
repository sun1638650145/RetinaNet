from tensorflow.keras import initializers
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def build_subnet(output_filters, bias_initializer='zeros', name=None):
    """构建功能子网络.

    Args:
        output_filters: int,
            功能子网络输出层的神经元数量.
        bias_initializer: str or tf.keras.initializers.Initializer instance, default='zeros',
            网络层偏置项初始化器.
        name: (可选) str, default=None, 功能子网络名称.

    Return:
        tf.keras.models.Sequential, 功能子网络实例.
    """
    model = Sequential(name=name)
    model.add(layers.InputLayer(input_shape=[None, None, 256]))  # 输入特征是每层FPN.

    _kernel_initializer = initializers.RandomNormal(mean=0.0, stddev=0.01)  # 高斯分布初始化.
    for _ in range(4):
        model.add(layers.Conv2D(filters=256,
                                kernel_size=[3, 3],
                                strides=(1, 1),
                                padding='same',
                                kernel_initializer=_kernel_initializer))
        model.add(layers.ReLU())
    model.add(layers.Conv2D(filters=output_filters,
                            kernel_size=[3, 3],
                            strides=(1, 1),
                            padding='same',
                            kernel_initializer=_kernel_initializer,
                            bias_initializer=bias_initializer))

    return model
