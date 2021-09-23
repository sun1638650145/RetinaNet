from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model


def get_backbone():
    """获取ResNet50的骨架网络."""
    backbone = ResNet50(include_top=False, input_shape=[None, None, 3])
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
    ]

    return Model(inputs=backbone.inputs, outputs=[c3_output, c4_output, c5_output])


class FeaturePyramidNetwork(layers.Layer):
    """特征金字塔网络."""
    def __init__(self):
        super(FeaturePyramidNetwork, self).__init__(name='FeaturePyramidNetwork')

        self.backbone = get_backbone()

        self.conv_c3_1x1 = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv_c4_1x1 = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')
        self.conv_c5_1x1 = layers.Conv2D(filters=256, kernel_size=1, strides=1, padding='same')

        self.conv_p3_3x3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.conv_p4_3x3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.conv_p5_3x3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')

        self.conv_c6_3x3 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')
        self.conv_p6_3x3 = layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same')

        self.upsample_2x = layers.UpSampling2D(size=2)

    def call(self, inputs, training=False, **kwargs):
        C3, C4, C5 = self.backbone(inputs, training=training)

        P5 = self.conv_c5_1x1(C5)
        P5 = self.conv_p5_3x3(P5)

        P4 = self.conv_c4_1x1(C4)
        P4 += self.upsample_2x(P5)
        P4 = self.conv_p4_3x3(P4)

        P3 = self.conv_c3_1x1(C3)
        P3 += self.upsample_2x(P4)
        P3 = self.conv_p3_3x3(P3)

        P6 = self.conv_c6_3x3(C5)
        P7 = self.conv_p6_3x3(relu(P6))  # 使用relu激活.

        return P3, P4, P5, P6, P7
