from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.image import ResizeMethod


def augment_data_builder(output_size: int = 512,
                         scale_min: float = 0.1,
                         scale_max: float = 2.0,
                         size_threshold: float = 0.01,
                         horizontal_flip_prob: float = 0.5,
                         resize_method: ResizeMethod = ResizeMethod.BILINEAR,
                         seed: Optional[int] = None):
    def _augment_data(sample):

        image = sample["image"]
        bboxes = sample["bboxes"]
        labels = sample["labels"]

        image, bboxes = random_flip_horizontal(image, bboxes, horizontal_flip_prob,
                                               seed)

        image, bboxes, labels = random_scale_crop_and_random_pad_to_square(
            image,
            bboxes,
            labels,
            scale_min,
            scale_max,
            output_size,
            size_threshold,
            resize_method,
        )

        return image, bboxes, labels

    return _augment_data


def random_flip_horizontal(image: tf.Tensor,
                           bboxes: tf.Tensor,
                           prob: float = 0.5,
                           seed: Optional[int] = None):
    """Flips image and boxes horizontally with 50% chance

    Extracted from: https://keras.io/examples/vision/retinanet/#preprocessing-data

    Horizontal flip:
     _________        _________
    |X        |      |        X|
    |         | -->  |         |
    |         |      |         |
    |_________|      |_________|

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` [ymin, xmin, ymax, xmax]
        representing bounding boxes, having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if prob > tf.random.uniform((), seed=seed):
        image = tf.image.flip_left_right(image)
        bboxes = tf.stack(
            [bboxes[:, 0], 1 - bboxes[:, 3], bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
    return image, bboxes


def random_scale_crop_and_random_pad_to_square(
    image: tf.Tensor,
    bboxes: tf.Tensor,
    labels: tf.Tensor,
    scale_min: float = 0.1,
    scale_max: float = 2.0,
    output_size: int = 512,
    size_threshold: float = 0.01,
    resize_method=tf.image.ResizeMethod.BILINEAR,
    seed: Optional[int] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    This augmentation function applies the following random augmentation steps to an
    input image and adjusts the ground truth respectively.

    1. random scale betwen scale_min and scale_max the image while preserving the aspect
        ratios
    2. random crop image to output size (if random scale factor is > 1.0)
    3. If one or both image dimensions are smaller than the output size the respective
        dimenison(s) are padded randomly with zeros. Hereby, the padding is only
        applied to one randomly chosen side of each axis. Hence, if the size of the
        vertical axis is below the output size the padding is applied above or below.
        The same applies to the horizontal dimension where the padding is applied either
        the left or right side randomly.

    The bounding boxes are adjusted accordingly. If an object is completly outside the
    augmented image or its size below the size thrshold it is removed.

    This augmenation is heavily inspired by:
    https://github.com/tensorflow/models/blob/c3b4fa95955172a810175ec8a363ec4966d38ec3/research/object_detection/core/preprocessor.py#L4273

    Args:
        image (tf.Tensor): image 3-D tensor
        bboxes (tf.Tensor): tensor of shape [N, 4] containg relative yx centroids
            [ymin, xmin, ymax, xmax]
        labels (tf.Tensor): tensor of shape [N,] containg the encoded labels
        scale_min (float, optional): minumum scale of input image before cropping and
            padding. Defaults to 0.1.
        scale_max (float, optional): maximum scale of input image before cropping and
            padding. Defaults to 2.0.
        output_size (int, optional): [description]. Defaults to 512.
        size_threshold (float, optional): if at least one side of an object's bounding
            box relative size is below this number after the augmentation steps, it is
            removed from the ground truth. Hence, this number is supposed to be between
            .0 (basiicaly, all objects kept) and approaching 1.0 where only big ones
            are kept. Defaults to .01.
        seed (int, optional): seed for tensorflow pseudo random number generator.
            Defaults to None.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: output image of size
            (output_size, output_size, ...), bounding boxes, labels
    """

    # get image shape
    img_shape = tf.shape(image)
    input_height, input_width = img_shape[0], img_shape[1]

    # resize image while preserving aspect ratios
    random_scale = tf.random.uniform([], scale_min, scale_max, seed=seed)
    max_input_dim = tf.cast(tf.maximum(input_height, input_width), tf.float32)

    input_ar_y = tf.cast(input_height, tf.float32) / max_input_dim
    input_ar_x = tf.cast(input_width, tf.float32) / max_input_dim

    scaled_height = tf.cast(random_scale * output_size * input_ar_y, tf.int32)
    scaled_width = tf.cast(random_scale * output_size * input_ar_x, tf.int32)

    image = tf.image.resize(
        image,
        size=(scaled_height, scaled_width),
        method=resize_method,
    )

    offset_height = tf.cast(scaled_height - output_size, tf.float32)
    offset_width = tf.cast(scaled_width - output_size, tf.float32)

    # if a side is below output_size, how much padding is needed ?
    pad_height = tf.cast(
        tf.constant(-1, dtype=tf.float32) * tf.minimum(offset_height, 0),
        tf.int32,
    )
    pad_width = tf.cast(
        tf.constant(-1, dtype=tf.float32) * tf.minimum(offset_width, 0),
        tf.int32,
    )

    # define random cropping offsets
    offset_height = tf.maximum(0.0, offset_height) * tf.random.uniform(
        [], 0, 1, seed=seed)
    offset_width = tf.maximum(0.0, offset_width) * tf.random.uniform(
        [], 0, 1, seed=seed)
    offset_height = tf.cast(offset_height, tf.int32)
    offset_width = tf.cast(offset_width, tf.int32)

    # crop image
    image = image[offset_height:offset_height + output_size,
                  offset_width:offset_width + output_size, :, ]

    # adjust bboxes to cropping
    offset_width_rel = tf.cast(offset_width / scaled_width, tf.float32)
    offset_height_rel = tf.cast(offset_height / scaled_height, tf.float32)

    cropped_image_shape = tf.shape(image)
    cropped_height, cropped_width = cropped_image_shape[0], cropped_image_shape[1]

    scale_cropped_height = tf.cast(scaled_height / cropped_height, tf.float32)
    scale_cropped_width = tf.cast(scaled_width / cropped_width, tf.float32)

    bboxes = bboxes - [
        offset_height_rel, offset_width_rel, offset_height_rel, offset_width_rel
    ]
    bboxes = bboxes * [
        scale_cropped_height,
        scale_cropped_width,
        scale_cropped_height,
        scale_cropped_width,
    ]

    # clip bboxes
    bboxes = tf.clip_by_value(bboxes, 0.0, 1.0)

    # if needed, add padding randomly and adjust bboxes accordingly
    if pad_height > 0:
        pad_height_up = pad_height * tf.random.uniform([], 0, 2, dtype=tf.int32)
        pad_height_down = pad_height - pad_height_up
        image = tf.pad(image,
                       paddings=[[pad_height_up, pad_height_down], [0, 0], [0, 0]])

        bboxes = bboxes * [(scaled_height / output_size), 1,
                           (scaled_height / output_size), 1]
        bboxes = bboxes + [(pad_height_up / output_size), 0,
                           (pad_height_up / output_size), 0]

    if pad_width > 0:
        pad_width_left = pad_width * tf.random.uniform([], 0, 2, dtype=tf.int32)
        pad_width_right = pad_width - pad_width_left

        image = tf.pad(
            image,
            paddings=[[0, 0], [pad_width_left, pad_width_right], [0, 0]],
        )
        bboxes = bboxes * [
            1, (scaled_width / output_size), 1, (scaled_width / output_size)
        ]
        bboxes = bboxes + [
            0, (pad_width_left / output_size), 0, (pad_width_left / output_size)
        ]

    # remove object completely outisde the augmented image
    bboxes, labels = remove_bboxes_outside(bboxes, labels)

    bboxes, lables = remove_small_bboxes(bboxes, labels, size_threshold)

    return image, bboxes, labels


def remove_bboxes_outside(bboxes: tf.Tensor, labels: tf.Tensor):
    """
    removes objects labels complete outside
    bbox format: relative [ymin, xmin, ymax, xmax]
    """
    violations = tf.stack(
        [
            tf.greater_equal(bboxes[:, 0], 1),
            tf.greater_equal(bboxes[:, 1], 1),
            tf.less_equal(bboxes[:, 2], 0),
            tf.less_equal(bboxes[:, 3], 0),
        ],
        axis=1,
    )

    keep = tf.logical_not(tf.reduce_any(violations, 1))
    bboxes = tf.boolean_mask(bboxes, keep, axis=0)
    labels = tf.boolean_mask(labels, keep, axis=0)

    return bboxes, labels


def remove_small_bboxes(bboxes: tf.Tensor, labels: tf.Tensor, threshold: float):
    """
    bbox format: relative [ymin, xmin, ymax, xmax]
    """
    keep = tf.stack(
        [
            tf.greater_equal(bboxes[:, 2] - bboxes[:, 0], threshold),
            tf.greater_equal(bboxes[:, 3] - bboxes[:, 1], threshold)
        ],
        axis=1,
    )

    keep = tf.reduce_all(keep, axis=1)

    bboxes = tf.boolean_mask(bboxes, keep, axis=0)
    labels = tf.boolean_mask(labels, keep, axis=0)

    return bboxes, labels
