import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class FocalLoss(Loss):
    """Focal损失函数.

    Attributes:
        alpha: float, default=0.75,
            权重因子, 用以解决类别不平衡.
        gamma: float, default=2.0,
            交叉熵的调制因子.

    References:
        - [Lin, T. Y. , et al., 2017](https://arxiv.org/abs/1708.02002v2)
    """
    def __init__(self, alpha=0.75, gamma=2.0, name='FocalLoss'):
        """初始化Focal损失函数.

        Args:
            alpha: float, default=0.75,
                权重因子, 用以解决类别不平衡.
            gamma: float, default=2.0,
                交叉熵的调制因子.
            name: (可选) str, default='FocalLoss', 自定义损失函数名称.
        """
        super(FocalLoss, self).__init__(reduction='none', name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        """调用Focal损失函数实例, 计算Focal损失函数值.

        Args:
            y_true: tf.Tensor or array-like, 真实值.
            y_pred: tf.Tensor or array-like, 预测值.

        Return:
            Focal损失函数值.
        """
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
