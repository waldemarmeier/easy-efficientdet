from typing import Tuple, Union

import tensorflow as tf

from easy_efficientdet._third_party.training import CosineLrSchedule
from easy_efficientdet.config import ObjectDetectionConfig
from easy_efficientdet.data.preprocessing import init_data
from easy_efficientdet.losses import ObjectDetectionLoss
from easy_efficientdet.model import EfficientDet
from easy_efficientdet.utils import DataSplit, setup_default_logger

logger = setup_default_logger("efficientdet-factory")


class EfficientDetFactory:
    def __init__(self, config: ObjectDetectionConfig):
        self.config = config

    def build_model(self) -> tf.keras.Model:
        return EfficientDet(**self.config.get_model_config())

    def init_data(
        self,
        data_split: Union[DataSplit, str],
        auto_train_data_size: bool = True
    ) -> Union[tf.data.Dataset, Tuple[tf.data.Dataset]]:

        if data_split in (DataSplit.TRAIN, DataSplit.TRAIN_VAL):
            if (not auto_train_data_size) and (self.config.train_data_size is None):
                logger.warning(
                    "Training data size is neither inferred nor set in config")

        if data_split == DataSplit.TRAIN_VAL:

            if self.config.train_data_path is not None \
                    and self.config.val_data_path is not None:
                train_data, val_data = init_data(self.config, data_split,
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

            train_data = init_data(self.config, DataSplit.TRAIN)
            if auto_train_data_size:
                _cardinality_num = \
                    train_data.cardinality().numpy() * self.config.batch_size
                self.config._update_train_data_size(_cardinality_num)
            return train_data
        elif data_split == DataSplit.VALIDATION:
            return init_data(self.config, DataSplit.VALIDATION)
        elif data_split == DataSplit.TEST:
            raise NotImplementedError("test data split is not implemented")

    def init_training(
            self) -> Tuple[tf.keras.optimizers.Optimizer, tf.keras.losses.Loss]:

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

        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                            momentum=self.config.momentum,
                                            decay=self.config.weight_decay)
        loss = ObjectDetectionLoss(**self.config.get_loss_config())

        return (optimizer, loss)
