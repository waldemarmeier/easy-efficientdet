# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================\
import math

import tensorflow as tf

from ..utils import setup_default_logger

_EFFICIENTDET_BATCH_SIZE = 128
_EFFICIENTDET_BASE_LEARNIN_RATE = .16
# is acutally .0, but don't want to waste time
_EFFICIENTDET_WARMUP_LEARNING_RATE = .0005


class CosineLrSchedule(tf.optimizers.schedules.LearningRateSchedule):
    """Cosine learning rate schedule."""

    # fix parameter names and get_config method stuff
    def __init__(self,
                 adjusted_lr: float,
                 lr_warmup_init: float,
                 lr_warmup_step: int,
                 total_steps: int = 0,
                 decay_steps=None):
        """Build a CosineLrSchedule.
        Args:
          adjusted_lr: `float`, The initial learning rate.
          lr_warmup_init: `float`, The warm up learning rate.
          lr_warmup_step: `int`, The warm up step.
          total_steps: `int`, Total train steps.
        """
        super().__init__()
        self.logger = setup_default_logger('CosineLrSchedule')
        self.adjusted_lr = adjusted_lr
        self.lr_warmup_init = lr_warmup_init
        self.lr_warmup_step = lr_warmup_step
        self.decay_steps = float(total_steps - lr_warmup_step)
        # TODO fix logging issues, add missing values
        values = f"\ntotal_steps: {total_steps}\nwarmup_steps: "
        "{self.warmup_steps}"
        values += f"\nadjusted_lr: {adjusted_lr}\nwarmup_learning_rate: "
        "{self.warmup_lr}"
        self.logger.info(
            f"Initializing WarmUpCosineDecayScheduler with following values:{values}")

    def __call__(self, step):
        linear_warmup = (self.lr_warmup_init +
                         (tf.cast(step, dtype=tf.float32) / self.lr_warmup_step *
                          (self.adjusted_lr - self.lr_warmup_init)))
        cosine_lr = 0.5 * self.adjusted_lr * (
            1 + tf.cos(math.pi * tf.cast(step, tf.float32) / self.decay_steps))
        return tf.where(step < self.lr_warmup_step, linear_warmup, cosine_lr)

    @classmethod
    def get_effdet_lr_scheduler(cls,
                                train_ds_size: int,
                                batch_size: int,
                                epochs: int,
                                warmup_epochs: int = 3):

        # e.g. if lr is 8 -> relative_bs=(8/128) or (1/16)
        relative_bs = batch_size / _EFFICIENTDET_BATCH_SIZE

        steps = int(train_ds_size / batch_size)
        total_steps = steps * epochs
        warmup_steps = warmup_epochs * steps

        # efficientdet paper uses huge batch sizes, decrease learning rates
        # proportionally to make sure, that training is stable
        base_lr = _EFFICIENTDET_BASE_LEARNIN_RATE * relative_bs
        warmup_lr = _EFFICIENTDET_WARMUP_LEARNING_RATE * relative_bs

        return cls(adjusted_lr=base_lr,
                   lr_warmup_init=warmup_lr,
                   lr_warmup_step=warmup_steps,
                   total_steps=total_steps)

    def get_config(self):
        return {
            "adjusted_lr": self.adjusted_lr,
            "lr_warmup_init": self.lr_warmup_init,
            "lr_warmup_step": self.lr_warmup_step,
            "decay_steps": self.decay_steps
        }
