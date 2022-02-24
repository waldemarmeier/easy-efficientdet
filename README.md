# easy-efficientdet

`easy-efficientdet` is an object detection package based on tensorflow which focuses on ease of use. Nevertheless, the layered API allows to easily replace the provided abstractions with custom code without having to write much boilerplate code.

Training object detection models can be overwhelming. This package aims to provide a one stop solution for
preprocessing (custom) labeled data, model training, evaluation and model optimzation for production.
This is accomplished by providing many abstractions over a simple to use API, reasonable default parameters and
transfer learning which speeds up training and improves model precision.

This project originated out of my master thesis which I wrote in coorporation with Detsche Bahn AG's (German railway company) House of AI. Deutsche Bahn generously allowed me to refactor and publish the code I wrote during my master thesis. If your are interested in working on innovative data and AI projects you might want get in touch with them.

## Features

- dataset preprocessing for COCO and PASCAL/VOC formats to TF-records for optimal training
- reasonable default parameters for training and many hints for unreasonable settings
- data augmentation
- set up of training routine inlcuding optimizer and learning rate schedule
- multi-gpu training
- quantization api for production tf-lite models **in development**

## Examples

- link to colab
- code

### Brief Introduction

```python
from easy_efficientdet import DefaultConfig, EfficientDet, EfficientDetFactory

# setup training
config = DefaultConfig(num_cls=20, 
                       batch_size=16, 
                       train_data_path='./data_train_tfr/', 
                       val_data_path='./data_val_tfr/', 
                       epochs=5)
factory = EfficientDetFactory(config)

# use configuration build model
model = factory.build_model()
# create data (train data includes augmentation steps)
data_train, data_val = factory.build_data_pipeline("train/val")

# optimizer, loss function and lr schedule an based on presets
opt = factory.create_optimizer()
loss = factory.create_loss_fn()

# keras api
model.compile(opt, loss)
model.fit(
    data_train,
    epochs=config.epochs,
    validation_data=data_val,
    verbose=1,
)
```

## Acknowledgements

Obviously, this package uses an EfficientDet model implementation for object detection. This project is based on research by Mingxing Tan, Ruoming Pang and Quoc V. Le [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070). I tried to replicate their implemenation as closely as possible. Additionally, I looked at the EfficientDet implementation in the [tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection) and the [RetinaNet tutorial from keras exampes](https://keras.io/examples/vision/retinanet/).

All thid party code that is used in this project is Apache 2.0 licensed and is located in `easy_efficientdet/_third_party/`. Naturally, the respective copyright headers have been kept in place (see beelow for a complete list).

In the meantime, the tensorflow team implemented a similiar package in [tensorflow example](https://github.com/tensorflow/examples). I took inspiration from it as well but no code is used directly. Finally, I used many of the training recommendations of [Ross Wightman](https://github.com/rwightman/efficientdet-pytorch).

### Third party code used in this project

- [tensorflow](https://github.com/tensorflow/tensorflow)
- [tensorflow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection)
- [keras RetinaNet example](https://keras.io/examples/vision/retinanet/)
- [original EfficientDet implementation](https://github.com/google/automl/tree/master/efficientdet)

## Known issues

- dynamic range and integer quantization do not to work properly. I have not found the reasen, yet.

## Backlog

- unit-testing **in progress**
- improve documentation **in progress**
- partial model training (heads -> bifpn -> backbone)
- develop faster training schedule which lowers the learning rate automatically based on a heuristic
- EMA training
- auto quantization during training / post training
- auto distillation
- other data sources (generator, tensorflow datasets, csv)
- auto anchor boxes settting based on data set and clustering algorithm
- auto augmentation parameters (need to come up with heuristic based on label distribution)
- custom combined Soft-NMS tensorflow op (might be fun)
- gradient checkpointing to increase batch or model size (might be fun)
- more backbone types (e.g. EfficientNetV2 in tf 2.8.0)
