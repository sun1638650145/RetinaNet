# import os
# os.environ["http_proxy"] = "http://127.0.0.1:41091"
# os.environ["https_proxy"] = "http://127.0.0.1:41091"
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow.keras.backend as K
from tensorflow.keras.applications.resnet import preprocess_input

from RetinaNet.preprocessing import img_ops, label_ops
from RetinaNet.utils import swap_xy
from RetinaNet.utils import xyxy_convert_xywh


def _preprocess_data(sample):
    """预处理数据."""
    image = sample['image']
    bboxes = swap_xy(sample['objects']['bbox'])  # 现在的bbox(xmin, ymin, xmax, ymax).
    labels = K.cast(sample['objects']['label'], dtype=tf.int32)

    # 数据增强, 随机水平翻转.
    image, bboxes = img_ops.random_flip_horizontal(image, bboxes)
    # ...
    image, img_shape, _ = img_ops.resize_and_pad_image(image)
    # 使用ImageNet的预处理方式处理图像.
    image = preprocess_input(image)

    # 还原归一化倍率.
    xmin = bboxes[..., 0] * img_shape[1]
    ymin = bboxes[..., 1] * img_shape[0]
    xmax = bboxes[..., 2] * img_shape[1]
    ymax = bboxes[..., 3] * img_shape[0]
    bboxes = K.stack([xmin, ymin, xmax, ymax], axis=-1)

    bboxes = xyxy_convert_xywh(bboxes)  # 现在的bbox(x, y, w, h).

    return image, bboxes, labels


def load_coco2017_dataset(data_dir, batch_size=2, split=('train', 'test')):
    """加载数据集."""
    train_dataset, test_dataset = tfds.load(name='coco/2017',
                                            split=split,
                                            data_dir=data_dir)

    _label_encoder = label_ops.LabelEncoder()

    # 只保留image, bbox, label.
    train_dataset = train_dataset.map(map_func=_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(batch_size=batch_size,
                                               padding_values=(0.0, 1e-8, -1),  # images, bboxes, labels.
                                               drop_remainder=True)
    # 处理label.
    train_dataset = train_dataset.map(map_func=_label_encoder.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
    # 进行预加载.
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # 同样的方式处理测试集.
    test_dataset = test_dataset.map(map_func=_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.shuffle(8 * batch_size)
    test_dataset = test_dataset.padded_batch(batch_size=1,
                                             padding_values=(0.0, 1e-8, -1),
                                             drop_remainder=True)
    test_dataset = test_dataset.map(map_func=_label_encoder.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset
