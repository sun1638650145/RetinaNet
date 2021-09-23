import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class FocalLoss(Loss):
    """Focal损失函数."""
    def __init__(self, alpha=0.75, gamma=2.0, name='FocalLoss'):
        super(FocalLoss, self).__init__(reduction='none', name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        probs = tf.nn.sigmoid(y_pred)
        pt = tf.where(condition=K.equal(y_true, 1.0),
                      x=probs,
                      y=1.0 - probs)

        at = tf.where(condition=K.equal(y_true, 1.0),
                      x=self.alpha,
                      y=1.0 - self.alpha)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                     logits=y_pred)

        loss = at * tf.pow(1.0 - pt, self.gamma) * ce

        return K.sum(loss, axis=-1)
