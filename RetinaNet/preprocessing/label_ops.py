import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from RetinaNet.utils import xywh_convert_xyxy


def compute_iou(boxes1, boxes2):
    """计算交并比(此计算方式仅作为原理展示, 时间复杂度和空间复杂度都过高, 请不要直接使用).

    Args:
        boxes1, boxes2: tf.Tensor, 边界框[x, y, width, height].

    Returns:
        边界框的交并比.
    """
    boxes1 = xywh_convert_xyxy(boxes1)
    x1min = boxes1[..., 0]
    y1min = boxes1[..., 1]
    x1max = boxes1[..., 2]
    y1max = boxes1[..., 3]

    boxes2 = xywh_convert_xyxy(boxes2)
    x2min = boxes2[..., 0]
    y2min = boxes2[..., 1]
    x2max = boxes2[..., 2]
    y2max = boxes2[..., 3]

    m, n = boxes1.shape[0], boxes2.shape[0]
    iou = np.zeros(shape=[m, n])

    for i in range(m):
        boxes1_area = (x1max[i] - x1min[i]) * (y1max[i] - y1min[i])
        for j in range(n):
            # 计算相交部分.
            ximin = K.maximum(x1min[i], x2min[j])
            yimin = K.maximum(y1min[i], y2min[j])
            ximax = K.minimum(x1max[i], x2max[j])
            yimax = K.minimum(y1max[i], y2max[j])

            w = K.maximum(0, K.abs(ximin - ximax))
            h = K.maximum(0, K.abs(yimin - yimax))
            intersection_area = K.maximum(w * h, 0.0)
            # 计算并集部分.
            boxes2_area = (x2max[j] - x2min[j]) * (y2max[j] - y2min[j])
            union_area = K.maximum(boxes1_area + boxes2_area - intersection_area, 1e-8)  # 避免除零.

            iou[i][j] = intersection_area / union_area
    iou = tf.convert_to_tensor(iou, dtype=tf.float32)

    return K.clip(iou, 0.0, 1.0)


def faster_compute_iou(boxes1, boxes2):
    """更快速的计算交并比,
    同时使用vectorization和矩阵广播优化时间复杂度和空间复杂度.

    Args:
        boxes1, boxes2: tf.Tensor, 边界框[x, y, width, height].

    Returns:
        边界框的交并比.

    Thanks:
        @Srihari Humbarwadi(https://github.com/srihari-humbarwadi)
    """
    boxes1_corners = xywh_convert_xyxy(boxes1)
    boxes2_corners = xywh_convert_xyxy(boxes2)

    # 计算相交部分.
    left_upper = K.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])  # 取两个边界框[xmin, ymin]的最大值, 相交面最小坐标.
    right_down = K.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])  # 取两个边界框[xmax, ymax]的最小值, 相交面最大坐标.
    intersection = K.maximum(0.0, right_down - left_upper)  # 坐标之差即为相交面的宽高.
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]

    boxes1_area = boxes1[:, 2] * boxes1[:, 3]  # box[x, y, width, height] width * height
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]

    # 计算并集部分.
    union_area = K.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )

    return K.clip(intersection_area / union_area, 0.0, 1.0)


class AnchorBox(object):
    """锚框.

    Attributes:
        aspect_ratios: list, 宽高比.
        scales: list, 锚框的缩放比.
        strides: list, 滑动步长, 特征图和输入的图相差的倍数.
        num_anchors: int, 锚框的数量.
        areas: list, 锚框的面积.
        anchor_dims: list, 所有锚框的尺寸(areas * aspect_ratios * scales).

    References:
        - [Lin, T. Y. , et al., 2017](https://arxiv.org/abs/1708.02002v2)
    """
    def __init__(self):
        """初始化锚框."""
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1/3, 2/3]]

        self.strides = [8, 16, 32, 64, 128]
        self.num_anchors = len(self.aspect_ratios) * len(self.scales)
        self.areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self.anchor_dims = self._compute_dims()

    def generate_anchors(self, image_height, image_width):
        """为特征金字塔的所有特征图生成锚框.

        Args:
            image_height: int,
                输入图像的长度.
            image_width: int,
                输入图像的宽度.

        Returns:
            tf.Tensor, 所有的锚框.
        """
        anchors = [
            self._generate_anchors(tf.math.ceil(image_height / self.strides[index]),
                                   tf.math.ceil(image_width / self.strides[index]),
                                   level)
            for index, level in enumerate(range(3, 8))  # P3, P4, P5, P6, P7
        ]

        return K.concatenate(anchors, axis=0)

    def _generate_anchors(self, feature_height, feature_width, level):
        """为给定等级特征图生成锚框.

        Args:
            feature_height: int,
                输入特征图的长度.
            feature_width: int,
                输入特征图的宽度.
            level: int,
                特征图的等级.

        Returns:
            tf.Tensor, 给定等级特征图所有的锚框.
        """
        # 遍历特征图生成锚框x坐标点和y坐标点(+0.5成为中心点).
        x = tf.range(feature_width, dtype=tf.float32) + 0.5
        y = tf.range(feature_height, dtype=tf.float32) + 0.5
        # 合并成坐标(按照步长进行放大).
        centers = K.stack(tf.meshgrid(x, y), axis=-1) * self.strides[level - 3]
        # 增加不同形状锚框数量的中心点(这里是9种).
        centers = K.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self.num_anchors, 1])
        # 获取不同特征层的不同形状锚框的宽高.
        dims = self.anchor_dims[level - 3]
        # 扩充到特征层的数量.
        dims = tf.tile(dims, [feature_height, feature_width, 1, 1])
        # 合并中心点和宽高(x, y, width, height).
        anchors = K.concatenate([centers, dims], axis=-1)

        return K.reshape(anchors, [feature_height * feature_width * self.num_anchors, 4])  # 展平.

    def _compute_dims(self):
        """计算所有的锚框的尺寸."""
        anchor_dims_all = list()
        for area in self.areas:  # 不同的面积(对应在不同的特征图上).
            anchor_dims = list()
            for ratio in self.aspect_ratios:  # 不同宽高比.
                # ratio = width / height; height * width = height ^ 2 * ratio = area.
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = K.reshape(K.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2])  # [1, 1]为宽高数占位.
                for scale in self.scales:  # 不同的缩放比.
                    anchor_dims.append(dims * scale)
            anchor_dims_all.append(K.stack(anchor_dims, axis=-2))

        return anchor_dims_all


class LabelEncoder(object):
    """标签编码器.

    Attributes:
        anchor_box: RetinaNet.preprocessing.label_ops.AnchorBox,
            锚框.
        box_variance: tf.Tensor,
            框方差, 用来增大损失(小于1), 便于计算梯度.
    """
    def __init__(self):
        """初始化标签编码器."""
        self.anchor_box = AnchorBox()
        self.box_variance = tf.convert_to_tensor([0.1, 0.1, 0.2, 0.2], dtype=tf.float32)

    def encode_batch(self, batch_images, gt_bboxes, label_ids):
        """编码每个批次的标签(包括分类框和回归框).

        Args:
            batch_images: tf.Tensor, 一个批次的图像.
            gt_bboxes: tf.Tensor, 该批次图像对应的真实边界框.
            label_ids: tf.Tensor, 一个批次的图像对应的原始标签.

        Returns:
            图像和对应编码后的标签.
        """
        img_shape = K.shape(batch_images)
        batch_size = img_shape[0]

        # 创建标签数组.
        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(img_shape, gt_bboxes[i], label_ids[i])
            labels = labels.write(i, label)

        return batch_images, labels.stack()

    def _encode_sample(self, image_shape, gt_bboxes, cls_ids):
        """编码一张图片的标签(包括分类和回归框).

        Args:
            image_shape: tf.Tensor, 图片的形状.
            gt_bboxes: tf.Tensor, 该图片对应的真实边界框.
            cls_ids: tf.Tensor, 该图片对应的原始标签.

        Returns:
            编码后的标签.
        """
        cls_ids = K.cast(cls_ids, dtype=tf.float32)
        anchor_boxes = self.anchor_box.generate_anchors(image_shape[1], image_shape[2])  # 此刻的anchor_boxes还没有label.

        # 获取anchor分类标签.
        max_iou_idx, positive_mask, ignore_mask = self._match_anchor_boxes(anchor_boxes, gt_bboxes)
        matched_gt_cls_ids = K.gather(cls_ids, max_iou_idx)  # 匹配出来用于对齐.
        cls_target = tf.where(condition=K.not_equal(positive_mask, 1.0), x=-1.0, y=matched_gt_cls_ids)  # 正例是类ID, 否则-1.
        cls_target = tf.where(condition=K.equal(ignore_mask, 1.0), x=-2.0, y=cls_target)  # 正例是类ID, 否则-2.
        cls_target = K.expand_dims(cls_target, axis=-1)

        # 获取anchor回归框.
        matched_gt_boxes = K.gather(gt_bboxes, max_iou_idx)
        box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)

        label = K.concatenate([cls_target, box_target], axis=-1)

        return label

    @staticmethod
    def _match_anchor_boxes(anchor_boxes, gt_bboxes, match_iou=0.5, ignore_iou=0.4):
        """基于交并比将锚框和真实框匹配.

        Args:
            anchor_boxes: RetinaNet.preprocessing.label_ops.AnchorBox,
                锚框.
            gt_bboxes: tf.Tensor,
                真实边界框.
            match_iou: float, default=0.5,
                标记为真实对象IoU阈值.
            ignore_iou: float, default=0.4,
                忽略对象IoU阈值.

        Returns:
            取出最大IoU对应的索引, 正例和忽略部分的标签.
        """
        # 计算IoU.
        iou_matrix = faster_compute_iou(anchor_boxes, gt_bboxes)
        # 取出最大IoU和对应的索引.
        max_iou = K.max(iou_matrix, axis=1)
        max_iou_idx = K.argmax(iou_matrix, axis=1)
        # 标记前景, 背景和忽略部分的标签.
        positive_mask = K.greater_equal(max_iou, match_iou)  # iou>=0.5正例.
        negative_mask = K.less(max_iou, ignore_iou)  # iou<0.4反例.
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))  # iou=0.4~0.5忽略.

        return max_iou_idx, K.cast(positive_mask, dtype=tf.float32), K.cast(ignore_mask, dtype=tf.float32)

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """计算框回归变换系数(原理同Fast R-CNN),
         模型将使用框回归变换系数作为训练数据.

        Args:
            anchor_boxes: RetinaNet.preprocessing.label_ops.AnchorBox,
                锚框.
            matched_gt_boxes: tf.Tensor, 匹配真实边界框.

        Returns:
            框回归变换系数.

        Notes: 变换系数(进行归一化), 能减少不同尺度的真实损失一致但是视觉直观差异大的情况; 更加容易梯度计算.
        """
        box_target = K.concatenate(
            [(matched_gt_boxes[..., :2] - anchor_boxes[..., :2]) / anchor_boxes[..., 2:],  # (gt_x - ac_x, gt_y - ac_y).
             tf.math.log(matched_gt_boxes[..., 2:] / anchor_boxes[..., 2:])],  # 显著缩放差异.
            axis=-1,
        )

        box_target /= self.box_variance

        return box_target
