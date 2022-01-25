import json
import traceback
from inspect import isgeneratorfunction
from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf

from easy_efficientdet._third_party.training import CosineLrSchedule
from easy_efficientdet.anchors import generate_anchor_boxes
from easy_efficientdet.config import ObjectDetectionConfig
from easy_efficientdet.data.preprocessing import (
    TFDATA_AUTOTUNE,
    build_data_pipeline,
    create_image_generator,
    load_tfrecords,
    parse_od_record,
)
from easy_efficientdet.inference import build_inference_model
from easy_efficientdet.losses import ObjectDetectionLoss
from easy_efficientdet.model import EfficientDet
from easy_efficientdet.quantization import ExportModel, OptimzationType, quantize
from easy_efficientdet.utils import (
    DataSplit,
    ImageDataGenertor,
    LabelMapType,
    setup_default_logger,
)

logger = setup_default_logger("efficientdet-factory")


class EfficientDetFactory:
    def __init__(self, config: ObjectDetectionConfig):
        self.config = config
        self._dist_strategy = None

    def reset_dist_strategy(self, ) -> None:
        self._dist_strategy = None

    @property
    def dist_strategy(self, ) -> tf.distribute.MirroredStrategy:
        logger.info("using mirrored strategy for mutli GPU training")
        if self._dist_strategy is None:
            self._dist_strategy = tf.distribute.MirroredStrategy()
            logger.info(f"created new mirrored strategy scope {self._dist_strategy}")
        else:
            logger.info("using existing mirrored strategy scope "
                        f"{self._dist_strategy}")
        return self._dist_strategy

    def build_model(self) -> tf.keras.Model:

        if self.config.multi_gpu:
            with self.dist_strategy.scope():
                return EfficientDet(**self.config.get_model_config())
        else:
            return EfficientDet(**self.config.get_model_config())

    def restore_from_checkpoint(
        self,
        model: tf.keras.Model,
        checkpoint_dir: str,
        mult_checkpoints_dir: bool = True,
    ) -> None:

        if mult_checkpoints_dir:
            path_latest_chpkt = tf.train.latest_checkpoint(checkpoint_dir)
            if path_latest_chpkt is None:
                raise Exception("No valid checkpoint found in directory "
                                f"{checkpoint_dir}")

        else:
            path_latest_chpkt = checkpoint_dir

        logger.info(f"using checkpoint with path {path_latest_chpkt}")

        if self.config.multi_gpu is True:
            with self.dist_strategy.scope():
                self._restore_checkpoint(model, path_latest_chpkt)
        else:
            self._restore_checkpoint(model, path_latest_chpkt)

    def _restore_checkpoint(
        self,
        model: tf.keras.Model,
        checkpoint_path: str,
    ) -> None:
        checkpoint = tf.train.Checkpoint(model)
        try:
            checkpoint.restore(checkpoint_path).assert_consumed()
        except AssertionError:
            logger.warning(traceback.format_exc())
            logger.warning("an error occurred during restore of checkpoint "
                           f"{checkpoint_path}. Usually, issues with "
                           "'save_counter' variable can be ignored.")

    def build_data_pipeline(
        self,
        data_split: Union[DataSplit, str],
        auto_train_data_size: bool = True,
    ) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset]]:

        if data_split in (DataSplit.TRAIN, DataSplit.TRAIN_VAL):
            if (not auto_train_data_size) and (self.config.train_data_size is None):
                logger.warning(
                    "Training data size is neither inferred nor set in config")

        if data_split == DataSplit.TRAIN_VAL:

            if self.config.train_data_path is not None \
                    and self.config.val_data_path is not None:
                train_data, val_data = build_data_pipeline(self.config, data_split,
                                                           auto_train_data_size)
            else:
                raise ValueError(f"For data split {data_split} 'train_data_path' and "
                                 "'val_data_path' properties have to be set")

            if auto_train_data_size:
                _cardinality_num = \
                    train_data.cardinality().numpy() * self.config.batch_size
                self.config._update_train_data_size(_cardinality_num)

            return train_data, val_data

        elif data_split == DataSplit.TRAIN:

            train_data = build_data_pipeline(self.config, DataSplit.TRAIN)
            if auto_train_data_size:
                _cardinality_num = \
                    train_data.cardinality().numpy() * self.config.batch_size
                self.config._update_train_data_size(_cardinality_num)
            return train_data
        elif data_split == DataSplit.VALIDATION:
            return build_data_pipeline(self.config, DataSplit.VALIDATION)
        elif data_split == DataSplit.TEST:
            raise NotImplementedError("test data split is not implemented")

    def build_data_eval(
        self,
        path: str = None,
        tfrecord_suffix: str = None,
    ) -> tf.data.Dataset:

        if path is None:
            path = self.config.val_data_path
        if tfrecord_suffix is None:
            tfrecord_suffix = self.config.tfrecord_suffix

        data = load_tfrecords(path, tfrecord_suffix)
        data = data.map(parse_od_record, TFDATA_AUTOTUNE)
        return data

    def create_optimizer(self, ) -> tf.keras.optimizers.Optimizer:

        if isinstance(self.config.learning_rate, float):
            logger.warning("Setting learning rate to a constant value is not "
                           "recommended")
            learning_rate = self.config.learning_rate
        elif self.config.learning_rate == "auto":
            if self.config.train_data_size is None:
                logger.warning("If learning rate is set to 'auto' training data size "
                               "should be known")
            learning_rate = CosineLrSchedule.get_effdet_lr_scheduler(
                self.config.train_data_size, self.config.batch_size, self.config.epochs)

        if self.config.multi_gpu:
            with self.dist_strategy.scope():
                optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                                    momentum=self.config.momentum,
                                                    decay=self.config.weight_decay)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                                momentum=self.config.momentum,
                                                decay=self.config.weight_decay)

        return optimizer

    def create_loss_fn(self, ) -> tf.keras.losses.Loss:
        return ObjectDetectionLoss(**self.config.get_loss_config())

    def create_anchor_boxes(self, ):
        return generate_anchor_boxes(**self.config.get_anchor_box_config())

    def load_labelmap(path: str) -> LabelMapType:
        with open(path) as fp:
            return json.load(fp)

    def quantize_model(
        self,
        model: tf.keras.Model,
        filename: str,
        score_thresh: float = .01,
        iou_thresh: float = .5,
        max_detections: int = 100,
        image_shape: Sequence[int] = None,
        opt_type: OptimzationType = OptimzationType.FLOAT32,
        representative_dataset: Optional[Union[DataSplit, ImageDataGenertor]] = None,
        size_limit: Optional[int] = None,
    ) -> bytes:

        if "decode" in model.layers[-1].name.lower():
            logger.warning("provide an object detection model without final "
                           "detection layer for NMS because this method "
                           "provides its own tflite compatible NMS implementation")

        if image_shape is None:
            image_shape = self.config.image_shape
            logger.info(f"Using image_shape {image_shape} from config")

        anchors = self.create_anchor_boxes()
        # normalize anchors
        anchors = anchors / [*self.config.image_shape[:2], *self.config.image_shape[:2]]
        export_model = ExportModel(self.config.num_cls, iou_thresh, score_thresh,
                                   max_detections, model, anchors)

        if opt_type not in OptimzationType:
            raise ValueError(f"opt_type {opt_type} is not. "
                             f"Should be in {OptimzationType.valid_types()}")

        quant_data = None
        if opt_type == OptimzationType.INT8:
            if isinstance(representative_dataset, str):
                if representative_dataset not in \
                        (DataSplit.TRAIN, DataSplit.VALIDATION):
                    raise ValueError("ony supports train or val datasets")

                if representative_dataset == DataSplit.TRAIN:
                    data_path = self.config.train_data_path
                elif representative_dataset == DataSplit.VALIDATION:
                    data_path = self.config.val_data_path

                quant_data = create_image_generator(data_path, image_shape[:2],
                                                    self.config.tfrecord_suffix,
                                                    size_limit)

            elif isgeneratorfunction(representative_dataset):
                quant_data = representative_dataset
            else:
                raise ValueError("For optimzation type int8 a representative_dataset "
                                 "has to be provided which is either a dataset "
                                 "from config (train/val) or a generator function "
                                 "for a dataset")

        logger.info(f"starting quantization of type {opt_type}")

        return quantize(export_model, opt_type, image_shape, quant_data, filename)

    @staticmethod
    def build_inference_model(
        model: tf.keras.Model,
        num_cls: int,
        image_shape: Sequence[int] = (512, 512, 3),
        confidence_threshold: float = 0.05,
        nms_iou_threshold: float = 0.5,
        max_detections_per_class: int = 100,
        max_detections: int = 100,
        box_variance: Optional[Sequence[float]] = None,
        resize: bool = False,
    ) -> tf.keras.Model:
        return build_inference_model(model, num_cls, image_shape, confidence_threshold,
                                     nms_iou_threshold, max_detections_per_class,
                                     max_detections, box_variance, resize)
