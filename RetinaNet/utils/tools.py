import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K


def swap_xy(bboxes):
    """交换x坐标和y坐标(TensorFlow Datasets的BBox的编码格式是[ymin, xmin, ymax, xmax]).

    Args:
        bboxes: tf.Tensor or array-like,
            编码格式是[ymin, xmin, ymax, xmax]的边界框.

    Return:
        bboxes: tf.Tensor or array-like,
            编码格式是[xmin, ymin, xmax, ymax]的边界框.
    """
    ymin = bboxes[..., 0]  # `...`比起`:`可以更轻松的处理1/2/3D张量.
    xmin = bboxes[..., 1]
    ymax = bboxes[..., 2]
    xmax = bboxes[..., 3]

    return K.stack([xmin, ymin, xmax, ymax], axis=-1)  # 2D->1;3D->2


def xyxy_convert_xywh(bboxes):
    """转换边界框编码.

    Args:
        bboxes: tf.Tensor or array-like,
            编码格式是[xmin, ymin, xmax, ymax]的边界框.

    Return:
        bboxes: tf.Tensor or array-like,
            编码格式是[x, y, width, height]的边界框.
    """
    x = (bboxes[..., 0] + bboxes[..., 2]) / 2.  # x = (xmin + xmax) / 2
    y = (bboxes[..., 1] + bboxes[..., 3]) / 2.  # y = (ymin + ymax) / 2
    w = bboxes[..., 2] - bboxes[..., 0]  # width = xmax - xmin
    h = bboxes[..., 3] - bboxes[..., 1]  # height = ymax - ymin

    return K.stack([x, y, w, h], axis=-1)


def xywh_convert_xyxy(bboxes):
    """转换边界框编码.

    Args:
        bboxes: tf.Tensor or array-like,
            编码格式是[x, y, width, height]的边界框.

    Return:
        bboxes: tf.Tensor or array-like,
            编码格式是[xmin, ymin, xmax, ymax]的边界框.
    """
    xmin = bboxes[..., 0] - bboxes[..., 2] / 2.  # xmin = x - width / 2
    xmax = bboxes[..., 0] + bboxes[..., 2] / 2.  # xmax = x + width / 2
    ymin = bboxes[..., 1] - bboxes[..., 3] / 2.  # ymin = y - height / 2
    ymax = bboxes[..., 1] + bboxes[..., 3] / 2.  # ymax = y + height / 2

    return K.stack([xmin, ymin, xmax, ymax], axis=-1)


def int2str(labels, decoding_dict):
    """转换整数标签为对应的类名.

    Args:
        labels: array-like, 整数标签.
        decoding_dict: ClassLabel.int2str,
            解码字典(包含整数和类名的映射关系).

    Return:
        类名组成的列表.
    """
    if len(labels) == 1:
        return decoding_dict(labels)
    else:
        labels_ = list()
        for label in labels:
            labels_.append(decoding_dict(int(label)))

        return labels_


def visualize_detections(image, bboxes, labels, scores=None):
    """可视化检测出的目标.

    Args:
        image: array-like, 检测的图像.
        bboxes: tf.Tensor or array-like, 检测出的目标的边界框.
        labels: list, 检测出的目标类名组成的列表.
        scores: numpy.ndarray, default=None,
            检测出的目标的预测概率, 如果值的`None`, 则默认预测概率是100%.
    """
    axes = plt.subplot()
    axes.axis('off')

    # 获取图像尺寸.
    img_shape = image.shape

    # 可视化图像.
    axes.imshow(image)

    # 检查置信度(如果没有将默认为1).
    if scores is None:
        scores = np.ones(len(labels), dtype=np.float32)

    for bbox, label, score in zip(bboxes, labels, scores):
        x, y, w, h = bbox
        # 获得xmin, ymin.
        xmin = x - w / 2
        ymin = y - h / 2
        # 还原归一化倍率.
        xmin *= img_shape[1]
        ymin *= img_shape[0]
        w *= img_shape[1]
        h *= img_shape[0]
        # 绘制锚框.
        patch = plt.Rectangle((xmin, ymin), width=w, height=h, fill=False, linewidth=1, color='blue')
        axes.add_patch(patch)
        # 显示标签.
        text = '{}: {:.0f}%'.format(label, score * 100)
        plt.scatter(xmin, ymin)
        axes.text(xmin, ymin, text, bbox={'facecolor': 'blue', 'alpha': 0.5}, color='white')

    plt.show()
