"""
extracted from keras examples
no explicit license notice is provided
Original Code:
https://keras.io/examples/vision/retinanet/#implementing-a-custom-layer-to-decode-predictions
"""

from numbers import Number
from typing import Optional, Sequence

import tensorflow as tf
from tensorflow import keras

from easy_efficientdet.anchors import generate_anchor_boxes
from easy_efficientdet.utils import convert_to_corners


class DecodePredictions(keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.

    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """
    def __init__(
            self,
            num_classes: int = 4,
            image_shape: Sequence[Number] = (640, 640),
            confidence_threshold: float = 0.05,
            nms_iou_threshold: float = 0.5,
            max_detections_per_class: int = 100,
            max_detections: int = 100,
            box_variance: Optional[
                Sequence[float]] = None,  # default should be none, [0.1, 0.1, 0.2, 0.2]
            **kwargs):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self.anchor_boxes = generate_anchor_boxes(image_shape)
        self.box_variance = box_variance

    def _decode_box_predictions(self, anchor_boxes, box_predictions):

        if self.box_variance is not None:
            boxes = box_predictions * self.box_variance
        else:
            boxes = box_predictions

        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, predictions):
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(self.anchor_boxes[None, ...],
                                             box_predictions)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
