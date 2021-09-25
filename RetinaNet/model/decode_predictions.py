import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from RetinaNet.preprocessing import AnchorBox


class DecodePredictions(layers.Layer):
    """解码预测值网络层, 将RetinaNet的预测值使用非极大值抑制解码成人类可读形式.

    Attributes:
        anchor_box: RetinaNet.preprocessing.label_ops.AnchorBox,
            锚框.
        box_variance: tf.Tensor,
            框方差, 用来增大损失(小于1), 便于计算梯度.
        max_detections_per_class: int, default=100,
            每类目标出现的最大数量.
        max_detections: int, default=100,
            图片上出现的目前的最大数量.
        nms_iou_threshold: float, default=0.5,
            使用非极大值抑制时的IoU阈值.
        confidence_threshold: float, default=00.5,
            样本置信度阈值.
    """
    def __init__(self,
                 max_detections_per_class=100,
                 max_detections=100,
                 nms_iou_threshold=0.5,
                 confidence_threshold=0.05,
                 **kwargs):
        """初始化解码预测值网络层.

        Args:
            max_detections_per_class: int, default=100,
                每类目标出现的最大数量.
            max_detections: int, default=100,
                图片上出现的目前的最大数量.
            nms_iou_threshold: float, default=0.5,
                使用非极大值抑制时的IoU阈值.
            confidence_threshold: float, default=00.5,
                样本置信度阈值.
        """
        super(DecodePredictions, self).__init__(**kwargs)

        self.anchor_box = AnchorBox()
        self.box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections
        self.nms_iou_threshold = nms_iou_threshold
        self.confidence_threshold = confidence_threshold

    def call(self, inputs, *args, **kwargs):
        """实例化解码预测值网络层.

        Args:
            inputs: tf.Tensor,
                输入网络层.
            args:
                predictions: tf.Tensor,
                    RetinaNet模型的预测输出值.

        Returns:
            经过非极大值抑制和解码后的张量列表.
        """
        predictions = args[0]

        img_shape = K.cast(K.shape(inputs), dtype=tf.float32)  # (BHWC)
        anchor_boxes = self.anchor_box.generate_anchors(img_shape[1], img_shape[2])

        cls_predictions = tf.nn.sigmoid(predictions[..., :-4])
        box_predictions = predictions[..., -4:]

        boxes = self._decode_box_predictions(K.expand_dims(anchor_boxes, axis=0), box_predictions)

        return tf.image.combined_non_max_suppression(
            boxes=K.expand_dims(boxes, axis=2),
            scores=cls_predictions,
            max_output_size_per_class=self.max_detections_per_class,
            max_total_size=self.max_detections,
            iou_threshold=self.nms_iou_threshold,
            score_threshold=self.confidence_threshold,
            clip_boxes=True,
        )

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        """根据框方差和框回归系数对回归框进行解码.

        Args:
            anchor_boxes: tf.Tensor, 锚框坐标信息.
            box_predictions: tf.Tensor, 模型预测回归框坐标信息.

        Return:
            tf.Tensor, 解码后的回归框.
        """
        boxes = box_predictions * self.box_variance  # 编码时使用框方差, 解码要恢复处理.
        boxes = K.concatenate(
            [
                boxes[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
                K.exp(boxes[..., 2:]) * anchor_boxes[..., 2:],
            ],
            axis=-1
        )

        return boxes
