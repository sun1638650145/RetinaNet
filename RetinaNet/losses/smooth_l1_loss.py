import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class SmoothL1Loss(Loss):
    """使用平滑的L1损失函数."""
    def __init__(self, delta=1.0, name='SmoothL1Loss'):
        super(SmoothL1Loss, self).__init__(reduction='none', name=name)
        self.delta = delta

    def call(self, y_true, y_pred):
        loss = y_true - y_pred

        absolute_loss = K.abs(loss)
        square_loss = K.square(loss)

        loss = tf.where(condition=K.less(absolute_loss, self.delta),
                        x=0.5 * square_loss,
                        y=absolute_loss - 0.5)

        return K.sum(loss, axis=-1)
