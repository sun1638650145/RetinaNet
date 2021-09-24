import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class SmoothL1Loss(Loss):
    """带平滑的L1损失函数.

    Attributes:
        delta: float, default=1.0, 平滑因子.
    """
    def __init__(self, delta=1.0, name='SmoothL1Loss'):
        """初始化带平滑的L1损失函数.

        Args:
            delta: float, default=1.0, 平滑因子.
            name: (可选) str, default='SmoothL1Loss', 自定义损失函数名称.
        """
        super(SmoothL1Loss, self).__init__(reduction='none', name=name)
        self.delta = delta

    def call(self, y_true, y_pred):
        """调用带平滑的L1损失函数实例, 计算带平滑的L1损失函数值.

        Args:
            y_true: tf.Tensor or array-like, 真实值.
            y_pred: tf.Tensor or array-like, 预测值.

        Return:
            带平滑的L1损失函数值.
        """
        loss = y_true - y_pred

        absolute_loss = K.abs(loss)
        square_loss = K.square(loss)

        loss = tf.where(condition=K.less(absolute_loss, self.delta),
                        x=0.5 * square_loss,
                        y=absolute_loss - 0.5)

        return K.sum(loss, axis=-1)
