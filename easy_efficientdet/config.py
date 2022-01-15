import dataclasses
import json
import os
from copy import deepcopy
from datetime import datetime
from numbers import Number
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union

import tensorflow as tf
from tensorflow.keras.losses import Reduction

from easy_efficientdet.utils import convert_image_to_rgb, setup_default_logger

# works better with linter
ResizeMethod = tf.image.ResizeMethod
logger = setup_default_logger("Config")

PREPROCESSING_FUNCS = {
    "tf.identity": tf.identity,
    "convert_image_to_rgb": convert_image_to_rgb
}


@dataclasses.dataclass
class ObjectDetectionConfig:

    # meta stuff
    file_name_template: ClassVar[str] = \
        "config_effdet_{version}_{timestamp}.json"
    date: str

    # need this in several places
    image_shape: Sequence[int]
    # model parameers
    efficientdet_version: int
    num_cls: int
    bn_sync: bool
    path_weights: Optional[str]
    # training (and later inference) parameters
    intermediate_scales: Union[Sequence[Number], int]
    aspect_ratios: Sequence[Number]
    stride_anchor_size_ratio: Number
    min_level: int
    max_level: int
    box_scales: Optional[Sequence[Number]]
    image_preprocessor: Union[Callable, str]
    match_iou: float
    ignore_iou: float
    # loss  parameters
    alpha: float
    gamma: float
    delta: float
    box_loss_weight: float
    reduction: Union[str, Reduction]
    # training
    batch_size: int
    learning_rate: Union[str, float]
    momentum: float
    weight_decay: float
    epochs: int
    warmup_epochs: int
    multi_gpu: bool
    # augmentation params
    scale_min: float
    scale_max: float
    training_image_size: int  # fit with image shape
    size_threshold: float  # relative size until which to keep the bounding boxes
    resize_method: ResizeMethod
    seed: Optional[int]
    horizontal_flip_prob: float
    # data stuff
    train_data_path: str
    train_data_size: Optional[int]
    val_data_path: str
    val_data_size: Optional[int]
    tfrecord_suffix: str

    def _update_train_data_size(self, train_data_size: int) -> None:
        if (train_data_size is not None) \
                and (train_data_size != tf.data.UNKNOWN_CARDINALITY):
            logger.info(f"'train_data_size' property is updated to {train_data_size}'")
            self.train_data_size = train_data_size
        else:
            logger.warning("Unable to infer training data size")

    @property
    def num_anchors(self) -> int:

        if isinstance(self.intermediate_scales, Sequence):
            intermediate_scales = len(self.intermediate_scales)
        else:
            intermediate_scales = self.intermediate_scales

        return intermediate_scales * len(self.aspect_ratios)

    def get_model_config(self) -> Dict[str, Any]:
        "return just the params needed to set up the model"
        return {
            "image_shape": deepcopy(self.image_shape),
            "version": self.efficientdet_version,
            "num_cls": self.num_cls,
            "num_anchors": self.num_anchors,
            "path_weights": self.path_weights,
            "multi_gpu": self.multi_gpu,
            "bn_sync": self.bn_sync,
        }

    def get_loss_config(self) -> Dict[str, Any]:
        "return params for setting up loss object"
        return {
            "num_cls": self.num_cls,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "delta": self.delta,
            "box_loss_weight": self.box_loss_weight,
            "reduction": self.reduction,
        }

    def get_augmentation_config(self) -> Dict[str, Any]:
        return {
            "output_size": self.training_image_size,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "size_threshold": self.size_threshold,
            "resize_method": self.resize_method,
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "seed": self.seed,
        }

    def get_encoding_config(self) -> Dict[str, Any]:
        return {
            "image_shape": self.image_shape,
            "intermediate_scales": self.intermediate_scales,
            "aspect_ratios": self.aspect_ratios,
            "stride_anchor_size_ratio": self.stride_anchor_size_ratio,
            "min_level": self.min_level,
            "max_level": self.max_level,
            "box_scales": self.box_scales,
            "image_preprocessor": self.image_preprocessor,
            "match_iou": self.match_iou,
            "ignore_iou": self.ignore_iou,
        }

    def save(self, file_path: Optional[str] = None) -> None:

        if file_path is None:
            file_path = self._get_default_file_name()
        elif (file_path[-1] in ['/', '\\']) or os.path.isdir(file_path):
            file_path = os.path.join(file_path, self._get_default_file_name())

        config: Dict[str, Any] = dataclasses.asdict(self)

        # image_preprocessor is callable (tf) function
        # replace it with its name
        if config["image_preprocessor"] is not None:
            image_preprocessor = config["image_preprocessor"]
            # check if this is a tf function, a bit hacky
            if hasattr(image_preprocessor, '_tf_decorator')\
               or hasattr(image_preprocessor, '_tf_dispatches'):
                image_preprocessor_name = image_preprocessor\
                                            .__wrapped__\
                                            ._tf_api_names[0]
                image_preprocessor_name = "tf." + image_preprocessor_name
                config["image_preprocessor"] = image_preprocessor_name
            else:
                image_preprocessor['image_preprocessor'] = image_preprocessor\
                                                            .__qualname__
            logger.info(
                "Cannot serialize image_preprocessor code of %s,"
                "only serializing name as string", config['image_preprocessor'])

        logger.info(f"serializing config to {file_path}")
        with open(file_path, 'w') as fp:
            json.dump(config, fp)

    @classmethod
    def load(path: str):
        # TODO implement loading
        raise NotImplementedError("loading not implemented yet")

    def _get_default_file_name(self) -> str:
        return self.file_name_template\
                .format(version=self.efficientdet_version,
                        timestamp=datetime.now().strftime('%Y%m%d%H%M'))


def DefaultConfig(num_cls: int,
                  batch_size: int,
                  train_data_path: str,
                  val_data_path: str,
                  epochs: int,
                  training_image_size: int = 512,
                  efficientdet_version: int = 0,
                  bw_image_data: bool = False,
                  **kwargs) -> ObjectDetectionConfig:
    # add here path to train
    default_config = {
        "date": datetime.now().strftime('%Y%m%d%H%M'),
        "path_weights": None,
        "intermediate_scales": 3,
        "aspect_ratios": [0.5, 1.0, 2.0],
        "stride_anchor_size_ratio": 4.0,
        "min_level": 3,
        "max_level": 7,
        "box_scales": None,
        "image_preprocessor": tf.identity,
        "match_iou": .5,
        "ignore_iou": .4,
        "alpha": .25,
        "gamma": 1.5,
        "delta": .1,
        "box_loss_weight": 50.0,
        "reduction": "auto",
        "scale_min": .1,
        "scale_max": 2.0,
        "size_threshold": .01,
        "resize_method": ResizeMethod.BILINEAR,
        "seed": None,
        "tfrecord_suffix": "tfrecord",
        "horizontal_flip_prob": .5,
        "train_data_size": None,
        "val_data_size": None,
        "learning_rate": "auto",
        "momentum": .9,
        "weight_decay": 4e-5,
        "warmup_epochs": 3,
        "multi_gpu": False,
        "bn_sync": False,
    }

    # get valid keys used later for validation of user input
    valid_kwargs_keys = frozenset(default_config.keys())

    default_config["training_image_size"] = training_image_size
    default_config["efficientdet_version"] = efficientdet_version
    default_config["batch_size"] = batch_size
    default_config["num_cls"] = num_cls
    default_config["train_data_path"] = train_data_path
    default_config["val_data_path"] = val_data_path
    default_config["epochs"] = epochs

    # multi-gpu training is set to true, use sync bn if not specified otherwise
    if "multi_gpu" in kwargs:
        if ("bn_sync" not in kwargs) and (kwargs["multi_gpu"] is True):
            logger.info("Using SyncBatchNormalitzation layers because multi-gpu "
                        "training is selected")
            default_config["bn_sync"] = True
        elif not kwargs["bn_sync"]:
            logger.warning("Multi-GPU training without SyncBatchNormalization is not"
                           " recommended. 'bn_sync' should be set to 'True'")

    if bw_image_data is True:
        default_config["image_shape"] = (training_image_size, training_image_size, 1)
        # efficient backbone expects input image with shape (..., 3)
        default_config["image_preprocessor"] = tf.image.grayscale_to_rgb
        if "image_preprocessor" in kwargs:
            logger.warning("Custom image preprocessor must convert BW images to RGB "
                           "with shape (..., 3), e.g. tf.image.grayscale_to_rgb")
    else:
        default_config["image_shape"] = (training_image_size, training_image_size, 3)

    if len(kwargs) > 0:
        # check if all kwargs are valid
        kwargs_fields = frozenset(kwargs.keys())
        invalid_keys = kwargs_fields.difference(valid_kwargs_keys)
        if len(invalid_keys) > 0:
            invalid_keys_list = ", ".join(invalid_keys)
            raise ValueError(f"Invalid Arguemnts: {invalid_keys_list}")
        else:
            for k, v in kwargs.items():
                default_config[k] = v

    _image_preprocessor = default_config["image_preprocessor"]

    if isinstance(_image_preprocessor, str):
        if _image_preprocessor in PREPROCESSING_FUNCS:
            default_config["image_preprocessor"] = \
                PREPROCESSING_FUNCS[_image_preprocessor]
        else:
            raise ValueError("Uknown 'image_preprocessor' function "
                             f"{_image_preprocessor}")

    if default_config["train_data_path"] is None:
        logger.warning("'train_data_path' has to be provided so set up "
                       "training automatically using Config object")

    if default_config["train_data_size"] is not None:
        logger.warning("training data size is set to "
                       f"{default_config['train_data_size']}. "
                       "Training data size has implications for automatic training "
                       "setup (learning rate schedule)). Make sure make sure "
                       "'train_data_size' is set correctly")

    if not (default_config["learning_rate"] == "auto"
            or isinstance(default_config["learning_rate"], float)):
        logger.warning("Learning rate should be set to 'auto' or a constant "
                       "float value")

    if default_config["warmup_epochs"] > epochs:
        logger.warning("Defined more warmup epochs than overall training epochs:"
                       f"{default_config['warmup_epochs']} vs. {epochs}")

    return ObjectDetectionConfig(**default_config)
