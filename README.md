# RetinaNet

没有什么可以多说的，就是使用Keras从零开始实现一个RetinaNet.

## 例子

1. 运行这段代码就可进行训练.

```python
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from RetinaNet import RetinaNetModel
from RetinaNet.losses import RetinaNetLoss
from RetinaNet.preprocessing import load_small_dataset


# 创建模型.
model = RetinaNetModel(num_classes=80)
model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=RetinaNetLoss(num_classes=80, alpha=0.25))
# 加载数据集.
train_dataset, test_dataset = load_small_dataset()
# 训练模型.
model.fit(x=train_dataset.take(100),
          epochs=100,
          verbose=1,
          callbacks=[
              ModelCheckpoint(
                  filepath='./checkpoint/',
                  monitor='val_loss',
                  verbose=1,
                  save_best_only=False,
                  save_weights_only=True,
              )
          ],
          validation_data=test_dataset.take(10))
```

2. 使用这段代码即可进行推理.

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications.resnet import preprocess_input

from RetinaNet import create_inference_model
from RetinaNet.preprocessing import resize_and_pad_image
from RetinaNet.utils import *


def prepare_image(image):
    """准备数据."""
    image, _, ratio = resize_and_pad_image(image, jitter=None)
    image = preprocess_input(image)

    return tf.expand_dims(image, axis=0), ratio


# 创建模型.
model = create_inference_model(weights_dir='./weights/', num_classes=80)
# 加载数据集.
val_dataset, dataset_info = tfds.load(name='coco/2017',
                                      split='validation',
                                      data_dir='./dataset/',
                                      with_info=True)

decoder = dataset_info.features['objects']['label'].int2str
sample = list(val_dataset.take(2).as_numpy_iterator())[0]

# 显示原图.
image = sample['image']
prepared_image, ratio = prepare_image(image)
prepared_image = tf.cast(prepared_image, dtype=tf.int32)
bboxes = swap_xy(sample['objects']['bbox'])
bboxes = xyxy_convert_xywh(bboxes)
labels = int2str(sample['objects']['label'], decoder)

visualize_detections(image, bboxes, labels)

# 显示推理结果.
detections = model.predict(prepared_image)
num_detections = detections.valid_detections[0]
detection_bboxes = detections.nmsed_boxes[0][:num_detections] / ratio
detection_labels = int2str(detections.nmsed_classes[0], decoder)
detection_scores = detections.nmsed_scores[0][:num_detections]

visualize_detections(image, detection_bboxes, detection_labels, detection_scores)
```

使用上面的两个基本代码模板就可以进行训练和推理，如果你想进行更加复杂的操作和更高精度训练准确率，你可以在模板的基础上进行修改，也可以自行编写.

## 其他问题

1. 目标检测需要的算力比较高，所以你需要一套强劲的GPU以及需要较长的训练时间，当然也可以在TPU上运行(P.S.作者有尝试，但是没成功).
2. 目前只能使用coco格式、tfrecord文件作为数据集载入.
3. [例子中使用的coco2017的子集.](https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip)

## 参考文献

* [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002v2)

* [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144v2)

