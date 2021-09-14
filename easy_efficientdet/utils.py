import logging
import sys
from numbers import Number
from typing import Sequence, Union

import tensorflow as tf


def key_value_getter(*args):
    def _key_value_getter(x):
        return {k: x[k] for k in args}

    return _key_value_getter


def setup_default_logger(name: str) -> logging.Logger:
    """Creates default logger which logs to stdout.

    Args:
        name (str): name of the logger

    Returns:
        logging.Logger: logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def convert_to_centroids(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes. To [x, y, w, h],
      being a centroid.
    """
    return tf.concat(
        [
            (boxes[..., :2] + boxes[..., 2:]) / 2.0,
            boxes[..., 2:] - boxes[..., :2],
        ],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format from centroids to corner coordinates.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]` (centroids).

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [
            boxes[..., :2] - boxes[..., 2:] / 2.0,
            boxes[..., :2] + boxes[..., 2:] / 2.0,
        ],
        axis=-1,
    )


def swap_xy(boxes: tf.Tensor) -> tf.Tensor:
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
      TODO improve language
      Obviuously, expects boxes to be in
        [xmin, ymin, xmax, ymax] -> [ymin, xmin, ymax, xmax]
      or [ymin, xmin, ymax, xmax] -> [xmin, ymin, xmax, ymax].

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack(
        [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]],
        axis=-1,
    )


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    TODO add notice where this func was extracted from

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(
        boxes1_corners[:, None, :2],
        boxes2_corners[:, :2],
    )
    rd = tf.minimum(
        boxes1_corners[:, None, 2:],
        boxes2_corners[:, 2:],
    )
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area,
        1e-8,
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def get_abs_bboxes(bboxes: tf.Tensor, image_shape: Sequence[Number]) -> tf.Tensor:
    """Converts bounding bboxes which are provided in relative corner xy format
     [xmin, ymin, xmax, ymax] to to absolute values for the given the image_shape.

    TODO change name to to_abs_bboxes

    Args:
        bboxes (tf.Tensor): bounding boxes in relative xy corner format
            [xmin, ymin, xmax, ymax] of shape (N, 4)
        image_shape (Sequence[Number]): shape of the respective image where the 0th
            entry contains the vertical size and 1st entry the horizontal one.

    Returns:
        tf.Tensor: bounding boxes in absolute corner xy coordinates
            [xmin, ymin, xmax, ymax] with shape (N, 4)
    """

    if len(bboxes.shape) > 2:  # TODO convert this to better tf-code using tf.shape
        raise ValueError(
            "Provided batched bounding boxes. Provide bounding boxes for a single "
            "image")

    bboxes = tf.stack(
        [
            bboxes[:, 0] * image_shape[1],
            bboxes[:, 1] * image_shape[0],
            bboxes[:, 2] * image_shape[1],
            bboxes[:, 3] * image_shape[0],
        ],
        axis=-1,
    )

    return bboxes


def get_tfds_size(tfds: tf.data.Dataset) -> int:
    @tf.function
    def _add_one(x, _):
        return x + 1

    return tfds.reduce(0, _add_one).numpy()


def infer_image_shape(image_size: Union[Sequence[Number], Number]) -> Sequence[Number]:

    if isinstance(image_size, (list, tuple)):
        if len(image_size) == 2:
            return image_size
        else:
            raise Exception(
                f"Image shape must be a sequence of length 2 not: {image_size}")
    else:
        return (image_size, image_size)
