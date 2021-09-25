import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import preprocess_input

from RetinaNet.preprocessing import img_ops, label_ops
from RetinaNet.utils import swap_xy
from RetinaNet.utils import xyxy_convert_xywh


def _preprocess_data(sample):
    """预处理数据;
     包括提取图像(对图像进行增强), 对应的边界框[ymin, xmin, ymax, xmax]和标签.

    Args:
        sample: dict of tf.Tensor, 样本数据组成的字典.

    Return:
        预处理后的图像, 对应的边界框[x, y, width, height]和标签.
    """
    image = sample['image']
    bboxes = swap_xy(sample['objects']['bbox'])  # 现在的bbox的编码格式[xmin, ymin, xmax, ymax].
    labels = K.cast(sample['objects']['label'], dtype=tf.int32)

    # 图像增强, 随机水平翻转.
    image, bboxes = img_ops.random_flip_horizontal(image, bboxes)
    # 调整并填充图像, 使得图像都能在最小特征图下被整除.
    image, img_shape, _ = img_ops.resize_and_pad_image(image)
    # 使用ImageNet的预处理方式处理图像.
    image = preprocess_input(image)

    # 还原归一化倍率.
    xmin = bboxes[..., 0] * img_shape[1]
    ymin = bboxes[..., 1] * img_shape[0]
    xmax = bboxes[..., 2] * img_shape[1]
    ymax = bboxes[..., 3] * img_shape[0]
    bboxes = K.stack([xmin, ymin, xmax, ymax], axis=-1)

    bboxes = xyxy_convert_xywh(bboxes)

    return image, bboxes, labels


def load_dataset(name='coco/2017',
                 dataset_dir=None,
                 batch_size=2,
                 split=('train', 'validation'),
                 without_preprocessing=False):
    """加载数据集(完整数据集).

    Args:
        name: str, default='coco/2017',
            使用的数据集名称, 数据集将从TensorFlow Datasets上下载(可能需要使用全局的代理).
        dataset_dir: str, default=None,
            如果该参数不是None, 将使用本地存储的路径加载数据集.
        batch_size: int, default=2,
            批次的大小.
        split: tuple or str, default=('train', 'validation'),
            数据集划分方式, 请使用tfds认可的三种数据划分方式['test', 'train', 'validation'].
        without_preprocessing: bool, default=False,
            不使用预处理(该参数只针对数据集划分为单一数据集的时候), 以及返回数据集信息.

    Returns:
        加载的数据集(可能对数据预处理), 数据集信息(仅在`without_preprocessing`为真的时候).

    Raises:
        ValueError: 未知划分方式, 请检查你的划分方式.
    """
    _label_encoder = label_ops.LabelEncoder()

    def _preprocess_dataset(_dataset):
        # 只保留image, bbox, label.
        _dataset = _dataset.map(map_func=_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
        _dataset = _dataset.shuffle(8 * batch_size)
        _dataset = _dataset.padded_batch(batch_size=batch_size,
                                         padding_values=(0.0, 1e-8, -1),  # images, bboxes, labels.
                                         drop_remainder=True)
        # 处理label.
        _dataset = _dataset.map(map_func=_label_encoder.encode_batch, num_parallel_calls=tf.data.AUTOTUNE)
        # 进行预加载.
        _dataset = _dataset.apply(tf.data.experimental.ignore_errors())
        _dataset = _dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return _dataset

    if type(split) is str:
        dataset, dataset_info = tfds.load(name=name, split=split, data_dir=dataset_dir, with_info=True)
        if without_preprocessing:
            return dataset, dataset_info

        return _preprocess_dataset(dataset)
    elif len(split) == 2:
        train_dataset, test_dataset = tfds.load(name=name,
                                                split=split,
                                                data_dir=dataset_dir)

        train_dataset = _preprocess_dataset(train_dataset)
        test_dataset = _preprocess_dataset(test_dataset)

        return train_dataset, test_dataset
    else:
        raise ValueError('未知划分方式, 请检查你的划分方式.')


def load_small_dataset(dataset_dir='./dataset/', batch_size=2, split=('train', 'validation')):
    """加载coco2017子数据集.

    Args:
        dataset_dir: str, default='./dataset/',
            请提前下载coco2017子数据集, 并建立对应路径保存.
            [coco2017子数据集](https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip)

        batch_size: int, default=2,
            批次的大小.
        split: tuple, default=('train', 'validation'),
            数据集划分方式, 请使用tfds认可的三种数据划分方式['test', 'train', 'validation'].

    Returns:
        加载的数据集.

    Raises:
        ValueError: 未知划分方式, 请检查你的划分方式.
    """
    return load_dataset(dataset_dir=dataset_dir, batch_size=batch_size, split=split)
