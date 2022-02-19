from dataclasses import dataclass
from numbers import Number
from typing import Dict, Sequence, Union

import tensorflow as tf


# TODO use this
@dataclass
class AnchorBoxesConfig:
    # TODO Change this to image shape object
    image_shape: Sequence[Number] = (512, 512)
    intermediate_scales: Union[Sequence[Number], int] = 3
    aspect_ratios: Sequence[Number] = [0.5, 1.0, 2.0]
    stride_anchor_size_ratio: Number = 4.0
    min_level: int = 3
    max_level: int = 7


def _get_featuremap_size(
    image_shape: Sequence[Number],
    min_level: int,
    max_level: int,
) -> Dict[int, Sequence[Number]]:
    """TODO

    Args:
        image_shape (Sequence[Number]): [description]
        min_level (int): [description]
        max_level (int): [description]

    Returns:
        Dict[int, Sequence[Number]]: [description]
    """
    feature_map_sizes = {}

    for level in range(min_level, max_level + 1):
        feature_map_sizes[level] = (
            image_shape[0] / (2**level),
            image_shape[1] / (2**level),
        )

    return feature_map_sizes


def _infer_intermediate_scales(
        intermediate_scales: Union[Sequence[Number], int]) -> Sequence[Number]:
    """[summary] todo

    Args:
        intermediate_scales (Union[Sequence[Number], int]): [description]

    Raises:
        Exception: [description]

    Returns:
        Sequence[Number]: [description]
    """
    if isinstance(intermediate_scales, int):
        return [2**(x / intermediate_scales) for x in range(intermediate_scales)]
    elif isinstance(intermediate_scales, Sequence):
        return intermediate_scales
    else:
        raise Exception(
            f"Unexpected 'intermediate_scales' parameter {intermediate_scales}. "
            "Expected int or Sequence of numbers.")


def generate_anchor_boxes(
    image_shape: Sequence[Number] = (512, 512),
    intermediate_scales: Union[Sequence[Number], int] = 3,
    aspect_ratios: Sequence[Number] = [0.5, 1.0, 2.0],
    stride_anchor_size_ratio: Number = 4.0,
    min_level: int = 3,
    max_level: int = 7,
) -> tf.Tensor:
    """Generate anchor boxes for given image shape and other parameters.
    @ TODO
        - reference the paper where anchor box generation is introduced
        - explain level, feature map and receptive field
        - for each level
        - intermediate scales for sizes between the discrete levels

    Args:
        image_shape (Sequence[Number], optional): Shape of the input image. Defaults to
             (512, 512).
        intermediate_scales (Union[Sequence[Number], int], optional): [description].
             Defaults to 3.
        aspect_ratios (Sequence[Number], optional): [description]. Defaults to
             [.5, 1.0, 2.0].
        stride_anchor_size_ratio (Number, optional): [description]. Defaults to 4.0.
        min_level (int, optional): [description]. Defaults to 3.
        max_level (int, optional): [description]. Defaults to 7.

    Returns:
        tf.Tensor: anchor boxes encoded as absolut centroids
             [x_center, y_center, width, height]
    """

    intermediate_scales = _infer_intermediate_scales(intermediate_scales)
    stride_anchor_size_ratio = tf.cast(
        tf.Variable(stride_anchor_size_ratio),
        tf.float32,
    )

    anchor_boxes = []

    anchor_boxes_per_coord = len(intermediate_scales) * len(aspect_ratios)

    feature_map_config = _get_featuremap_size(image_shape[:2], min_level, max_level)

    for level, feature_map in feature_map_config.items():

        feature_map = tf.cast(tf.Variable(feature_map), tf.float32)

        stride = 2**level
        base_anchor_size = stride * stride_anchor_size_ratio
        # TODO refactor should be named differently, e.g. base_sizes
        base_anchor_boxes = []

        for ratio in aspect_ratios:
            for scale in intermediate_scales:
                height = base_anchor_size * tf.sqrt(1 / ratio) * scale
                width = base_anchor_size * tf.sqrt(ratio) * scale
                base_anchor_boxes.append(tf.Variable([width, height], dtype=tf.float32))

        center_x = tf.range(stride / 2, image_shape[1], stride, dtype=tf.float32)
        center_y = tf.range(stride / 2, image_shape[0], stride, dtype=tf.float32)

        center_x, center_y = tf.meshgrid(center_x, center_y)

        center_x = tf.expand_dims(center_x, -1)
        center_y = tf.expand_dims(center_y, -1)

        center_x = tf.tile(center_x, (1, 1, anchor_boxes_per_coord))
        center_y = tf.tile(center_y, (1, 1, anchor_boxes_per_coord))

        center_x = tf.reshape(center_x, (-1, ))
        center_y = tf.reshape(center_y, (-1, ))

        center_coords = tf.stack([center_x, center_y], -1)

        base_anchor_boxes = tf.stack(base_anchor_boxes)
        base_anchor_boxes = tf.tile(base_anchor_boxes,
                                    (feature_map[0] * feature_map[1], 1))
        base_anchor_boxes = tf.concat([center_coords, base_anchor_boxes], -1)

        anchor_boxes.append(base_anchor_boxes)

    anchor_boxes = tf.concat(anchor_boxes, 0)

    return anchor_boxes
