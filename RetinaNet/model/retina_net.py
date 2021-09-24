import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from RetinaNet.model.decode_predictions import DecodePredictions
from RetinaNet.model.feature_pyramid_network import FeaturePyramidNetwork
from RetinaNet.model.subnet import build_subnet


class RetinaNetModel(Model):
    """RetinaNet模型,
    RetinaNet的网络结构是在ResNet50上生成一个FPN, FPN后连接分类和框回归两个功能子网络.

    Attributes:
        num_classes: int,
            训练数据目标类别总数.
        fpn: RetinaNet.model.feature_pyramid_network.FeaturePyramidNetwork,
            特征金字塔网络.
        clf_subnet: tf.keras.models.Sequential,
            分类子网络.
        box_subnet: tf.keras.models.Sequential,
            框回归子网络.

    References:
        - [Lin, T. Y. , et al., 2017](https://arxiv.org/abs/1708.02002v2)
    """
    def __init__(self, num_classes, num_anchors=9):
        """初始化RetinaNet模型.

        Args:
            num_classes: int,
                训练数据目标类别总数.
            num_anchors: int, default=9,
                锚框类型总数.
        """
        super(RetinaNetModel, self).__init__(name='RetinaNetModel')
        _clf_bias_initializer = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

        self.num_classes = num_classes

        self.fpn = FeaturePyramidNetwork()
        self.clf_subnet = build_subnet(self.num_classes * num_anchors,
                                       bias_initializer=_clf_bias_initializer,
                                       name='ClassificationSubnet')
        self.box_subnet = build_subnet(4 * num_anchors, bias_initializer='zeros', name='BoxRegressionSubnet')
        
    def call(self, inputs, training=None, mask=None):
        """实例化RetinaNet模型.

        Args:
            inputs: tf.Tensor,
                输入网络层.
            training: bool, default=None,
                网络是否可训练.
            mask: a mask or list of mask, default=None,
                掩码列表(这里没有使用到).

        Returns:
            tf.Tensor, RetinaNet的输出, 其中张量的最后四位为框回归预测参数, 剩余位数为分类独热编码预测概率.
        """
        features = self.fpn(inputs, training=training)
        num_images = tf.shape(inputs)[0]

        clf_outputs = list()
        box_outputs = list()

        for feature in features:
            clf_outputs.append(tf.reshape(self.clf_subnet(feature), shape=[num_images, -1, self.num_classes]))
            box_outputs.append(tf.reshape(self.box_subnet(feature), shape=[num_images, -1, 4]))

        clf_outputs = tf.concat(clf_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)

        return tf.concat([clf_outputs, box_outputs], axis=-1)

    def get_config(self):
        pass


def create_inference_model(weights_dir, num_classes, num_anchors=9):
    """创建RetinaNet模型, 在推理模式下工作.

    Args:
        weights_dir: str,
            RetinaNet的模型权重文件.
        num_classes: int,
            训练数据目标类别总数.
        num_anchors: int, default=9,
            锚框类型总数.

    Return:
        tf.keras.models.Model, RetinaNet模型实例.
    """
    image = layers.Input(shape=[None, None, 3], name='image')
    model = RetinaNetModel(num_classes, num_anchors)
    model.load_weights(weights_dir)

    predictions = model(image, training=False)
    detections_layer = DecodePredictions()(image, predictions)

    return Model(inputs=image, outputs=detections_layer)
