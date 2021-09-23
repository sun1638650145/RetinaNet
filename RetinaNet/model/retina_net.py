import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from RetinaNet.model.decode_predictions import DecodePredictions
from RetinaNet.model.feature_pyramid_network import FeaturePyramidNetwork
from RetinaNet.model.subnet import build_subnet


class RetinaNetModel(Model):
    """`RetinaNet`的网络结构是在`ResNet`上生成一个`FPN`,
    `FPN`附上分类和框两个子网络.
    """
    def __init__(self, num_classes, num_anchors=9):
        super(RetinaNetModel, self).__init__(name='RetinaNetModel')
        _cls_bias_initializer = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))

        self.num_classes = num_classes

        self.fpn = FeaturePyramidNetwork()
        self.cls_subnet = build_subnet(self.num_classes * num_anchors,
                                       bias_initializer=_cls_bias_initializer,
                                       name='ClassificationSubnet')
        self.box_subnet = build_subnet(4 * num_anchors, bias_initializer='zeros', name='BoxRegressionSubnet')
        
    def call(self, inputs, training=None, mask=None):
        features = self.fpn(inputs, training=training)
        num_images = tf.shape(inputs)[0]

        cls_outputs = list()
        box_outputs = list()

        for feature in features:
            cls_outputs.append(tf.reshape(self.cls_subnet(feature), shape=[num_images, -1, self.num_classes]))
            box_outputs.append(tf.reshape(self.box_subnet(feature), shape=[num_images, -1, 4]))

        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)

        return tf.concat([cls_outputs, box_outputs], axis=-1)


def create_inference_model(weights_dir, num_classes, num_anchors=9):
    """创建一个RetinaNet推理模型."""
    image = layers.Input(shape=[None, None, 3], name='image')
    model = RetinaNetModel(num_classes, num_anchors)
    model.load_weights(weights_dir)

    predictions = model(image, training=False)
    detections_layer = DecodePredictions()(image, predictions)

    return Model(inputs=image, outputs=detections_layer)
