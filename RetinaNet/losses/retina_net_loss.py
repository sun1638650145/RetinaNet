import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses

from RetinaNet.losses.focal_loss import FocalLoss
from RetinaNet.losses.smooth_l1_loss import SmoothL1Loss


class RetinaNetLoss(losses.Loss):
    """合并L1损失函数和Focal损失函数."""
    def __init__(self, num_classes=80, alpha=0.75, gamma=2.0, delta=1.0, name='RetinaNetLoss'):
        super(RetinaNetLoss, self).__init__(name=name)

        self.num_classes = num_classes

        self.clf_loss = FocalLoss(alpha, gamma, name='ClassificationLoss')
        self.box_loss = SmoothL1Loss(delta, name='BoxRegressionLoss')

    def call(self, y_true, y_pred):
        y_pred = K.cast(y_pred, dtype=tf.float32)  # default:80+4(cls+box).
        # 获取mask.
        positive_mask = K.cast(K.greater(y_true[..., 0], -1.0), dtype=tf.float32)
        ignore_mask = K.cast(K.equal(y_true[..., 0], -2.0), dtype=tf.float32)
        normalizer = K.sum(positive_mask, axis=-1)

        # 将类别标签转换为独热编码.
        cls_labels = tf.one_hot(
            indices=K.cast(y_true[..., 0], dtype=tf.int32),  # y_true:1+4(cls+box).
            depth=self.num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[..., :-4]
        # 计算分类损失.
        clf_loss = self.clf_loss(y_true=cls_labels, y_pred=cls_predictions)
        clf_loss = tf.where(condition=K.equal(ignore_mask, 1.0), x=0.0, y=clf_loss)  # 只要是不是忽略, 必然是正(反)例, 有损失.
        clf_loss = tf.math.divide_no_nan(K.sum(clf_loss, axis=-1), y=normalizer)

        # 计算回归损失.
        box_labels = y_true[..., 1:]
        box_predictions = y_pred[..., -4:]
        box_loss = self.box_loss(y_true=box_labels, y_pred=box_predictions)
        box_loss = tf.where(condition=K.equal(positive_mask, 1.0), x=box_loss, y=0.0)  # 只有框在正例上才有意义, 除此之外不应该有损失值.
        box_loss = tf.math.divide_no_nan(K.sum(box_loss, axis=-1), y=normalizer)

        loss = clf_loss + box_loss

        return loss
