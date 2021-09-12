from typing import Any, Dict, Union

import tensorflow as tf
from tensorflow.keras.losses import Reduction


class ObjectDetectionLoss(tf.losses.Loss):
    def __init__(
        self,
        num_cls: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        delta: float = 0.1,
        box_loss_weight: float = 50.0,
        reduction: Union[str, Reduction] = "auto",
        name: str = "combined_object_detection_loss",
    ):
        """Combined loss function for object detection. It combines huber loss for
        box regression and focal loss for classification.

        Details:
        For calculating the box loss only 'postive' anchors are taken into account.
        For the classification loss only 'postive' and 'negative' anchors are considered
        while 'ignored' anchors are removed from the calculation.
        For more details on the ground truth encoding have a look on the encoder class
        implementation.

        This implementation is highly inspired by:
        https://keras.io/examples/vision/retinanet/#implementing-smooth-l1-loss-and-focal-loss-as-keras-custom-losses

        Args:
            num_cls (int): Number of classes
            alpha (float, optional): weight parameter for focal loss. Defaults to 0.25.
            gamma (float, optional): focusing parameter for focal loss. Defaults to 2.0.
            delta (float, optional): delta parameter for huber loss function. Defaults
                to .1.
            box_loss_weight (float, optional): weighting factor for box loss. Defaults
                to 50.0.
            reduction (Union[str, tf.keras.losses.Reduction], optional): reduction for
                loss results. Defaults to "auto".
            name (str, optional): name for this loss function. Defaults to
                "combined_object_detection_loss".
        """
        super().__init__(
            reduction=reduction,
            name=name,
        )
        self._alpha = alpha
        self._gamma = gamma
        self._delta = delta
        self._num_cls = num_cls
        self._box_loss_weight = box_loss_weight
        self._huber_loss = tf.keras.losses.Huber(
            delta=delta,
            reduction=Reduction.SUM,
        )
        # todo label smoothing
        # todo find out if one can avoid using bboxes labeled as ignore
        #   and just set treat them as negatives e.g. setting
        #   targets to 0s, but for now they work just fine

    def get_config(self) -> Dict[str, Any]:
        # includes self.name & self.reduction
        default_config = super().get_config()
        custom_config = {
            "alpha": self._alpha,
            "gamma": self._gamma,
            "delta": self._delta,
            "num_cls": self._num_cls,
            "box_loss_weight": self._box_loss_weight
        }
        custom_config.update(default_config)
        return custom_config

    def box_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        normalizer: tf.Tensor,
        positive_mask: tf.Tensor,
    ) -> tf.Tensor:
        """Implements huber loss for regression with. The delta paramter is set
        on the class.

        Args:
            y_true (tf.Tensor): ground truth tensor with shape (B, N, 4)
            y_pred (tf.Tensor): predicted values with shape (B, N, 4)
            normalizer (tf.Tensor): normalizer skalar
            positive_mask (tf.Tensor): tensor of shape (B, N,) containing
                boolean or {0, 1} values. It indicates which values to
                to keep for calculating loss value. Usually, 'positive'
                and 'negative' values are kept while 'ignore' values
                are set to false.

        Returns:
            tf.Tensor:
        """
        # do not multiply the nomalizer * 4.0 because the keras
        # implementation of huber loss caluclates the mean for
        # every bounding box regression making it:
        #   loss[b, i] = (1/4) * sum(loss[b, i, :])
        # where b indicates the batch and i the index of the
        # anchor box. For a discussion on this topic see:
        # https://github.com/tensorflow/tensorflow/issues/27190

        y_true = tf.boolean_mask(y_true, positive_mask)
        y_pred = tf.boolean_mask(y_pred, positive_mask)
        box_loss = self._huber_loss(y_true, y_pred)
        box_loss = tf.math.divide_no_nan(box_loss, normalizer)
        return box_loss

    def clf_loss(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        normalizer: tf.Tensor,
        ignore_mask: tf.Tensor,
    ) -> tf.Tensor:
        """Classification loss function which implements focal loss. The alpha and
        gamma parameters are defined at class level.

        Args:
            y_true (tf.Tensor): one hot encoded ground truth tensor of shape (B, N, C)
            y_pred (tf.Tensor): predicted values tensor of shape (B, N, C)
            normalizer (tf.Tensor): skalar value for normalization
            ignore_mask (tf.Tensor): boolean tensor of shape (B, N,) indicating
                anchor box predicitons to ignore

        Returns:
            tf.Tensor: classification loss values
        """

        # alpha_t * (1 - p_t)**gamma * ce
        # use it for better numerical stability and abstracting
        # away tedious code, same as
        #   - y_true * log(sigm(y_pred)) - (1 - y_true) * log(sigm(y_pred))
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
        )

        y_pred_probs = tf.sigmoid(y_pred)
        alpha_t = tf.where(
            tf.equal(y_true, 1.0),
            self._alpha,
            1.0 - self._alpha,
        )
        p_t = tf.where(
            tf.equal(y_true, 1.0),
            y_pred_probs,
            1.0 - y_pred_probs,
        )

        focal_loss = alpha_t * tf.math.pow(1.0 - p_t, self._gamma) * cross_entropy
        # filter out ignored stuff
        keep_mask = tf.logical_not(ignore_mask)

        # remove loss values that are ignored
        # influence on learning / gradient:
        # -> positive matches: towards correct class
        # -> negative matches: towards no class / background class
        # -> ignored matches: no influence on gradient
        focal_loss = tf.boolean_mask(focal_loss, keep_mask)
        focal_loss = tf.math.divide_no_nan(focal_loss, normalizer)

        return focal_loss

    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
    ) -> tf.Tensor:
        """Calculates object detection loss given y_true and y_pred.
        The ground truth classes are expected to be nominally encoded where
        integer values >=0 indicate classes, -1 negative matches and
        -2 ignored ones. Since the resulting loss values have different
        value ranges the box loss is multiplied by a factor to compensate
        for that. The background class is encoded as vector of zeros
        are 'base class' in the context of one hot encoding.

        Args:
            y_true (tf.Tensor): ground truth values containing box and class targets of
                shape (B, N, 4 + 1)
            y_pred (tf.Tensor): predicted values tensor of shape (B, N, 4 + C)

        Returns:
            tf.Tensor: loss value
        """
        y_true_box = y_true[..., :4]
        y_pred_box = y_pred[..., :4]

        y_true_cls = y_true[..., 4]

        num_positives_batch = tf.reduce_sum(tf.cast(y_true_cls >= 0, tf.float32))
        # num positives in batch plus 1 for numerical stability
        if num_positives_batch == 0:
            num_positives_batch += 1

        y_true_cls = tf.one_hot(
            tf.cast(y_true_cls, tf.int32),
            depth=self._num_cls,
            dtype=tf.float32,
            name="one_hot_encode_cls",
        )
        y_pred_cls = y_pred[..., 4:]

        # get mask of values which are either positve or negative matches
        ignore_mask = tf.equal(y_true[..., 4], -2.0)
        positive_mask = tf.greater_equal(y_true[..., 4], 0.0)

        box_loss = self.box_loss(
            y_true_box,
            y_pred_box,
            normalizer=num_positives_batch,
            positive_mask=positive_mask,
        )

        clf_loss = self.clf_loss(
            y_true_cls,
            y_pred_cls,
            normalizer=num_positives_batch,
            ignore_mask=ignore_mask,
        )

        box_loss = tf.reduce_sum(box_loss)
        clf_loss = tf.reduce_sum(clf_loss)
        return box_loss * self._box_loss_weight + clf_loss
