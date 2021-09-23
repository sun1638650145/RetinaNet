import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K


def random_flip_horizontal(image, bboxes):
    """水平随机翻转."""
    if np.random.uniform() > 0.5:
        image = tf.image.flip_left_right(image)

        # 水平翻转纵坐标不变; 横坐标翻转xmin = 1 - xmax(归一化尺度上的).
        xmin_ = bboxes[..., 0]
        ymin = bboxes[..., 1]
        xmax_ = bboxes[..., 2]
        ymax = bboxes[..., 3]
        xmin = 1 - xmax_
        xmax = 1 - xmin_

        bboxes = K.stack([xmin, ymin, xmax, ymax], axis=-1)

    return image, bboxes


def resize_and_pad_image(image, jitter=(640, 1024), min_side=800.0, max_side=1333.0, stride=128.0):
    """保证原始横纵比的情况下, 调整并填充图像."""
    img_shape = K.cast(K.shape(image)[:2], dtype=tf.float32)  # 不需要色彩通道.
    # 随机抖动.
    if jitter:
        min_side = tf.random.uniform(shape=[], minval=jitter[0], maxval=jitter[1], dtype=tf.float32)
    # 获取倍率.
    ratio = min_side / tf.reduce_min(img_shape)
    if ratio * K.max(img_shape) > max_side:
        ratio = max_side / K.max(img_shape)
    # 重设图像尺寸和图像.
    img_shape *= ratio
    image = tf.image.resize(image, size=K.cast(img_shape, dtype=tf.int32))
    # 用零填充图像到指定尺寸.
    padded_img_shape = K.cast(tf.math.ceil(img_shape / stride) * stride, dtype=tf.int32)
    image = tf.image.pad_to_bounding_box(image, 0, 0, padded_img_shape[0], padded_img_shape[1])

    return image, img_shape, ratio
