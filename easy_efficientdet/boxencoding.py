from numbers import Number
from typing import Callable, Optional, Sequence, Tuple, Union

import tensorflow as tf

from easy_efficientdet.anchors import generate_anchor_boxes
from easy_efficientdet.utils import (
    compute_iou,
    convert_to_centroids,
    get_abs_bboxes,
    swap_xy,
)

EPSILON = 1e-8


class BoxEncoder:
    def __init__(
        self,
        image_shape: Sequence[Number] = (640, 640),
        intermediate_scales: Union[Sequence[Number], int] = 3,
        aspect_ratios: Sequence[Number] = [0.5, 1.0, 2.0],
        stride_anchor_size_ratio: Number = 4.0,
        min_level: int = 3,
        max_level: int = 7,
        box_scales: Optional[Sequence[Number]] = None,
        image_preprocessor: Callable = tf.identity,
        match_iou: float = 0.5,
        ignore_iou: Optional[float] = 0.4,
    ):
        self.image_shape = image_shape
        self.intermediate_scales = intermediate_scales
        self.aspect_ratio = aspect_ratios
        self.stride_anchor_size_ratio = stride_anchor_size_ratio
        self.min_level = min_level
        self.max_level = max_level
        self._anchors = generate_anchor_boxes(
            image_shape,
            intermediate_scales,
            aspect_ratios,
            stride_anchor_size_ratio,
            min_level,
            max_level,
        )
        self.box_scales = box_scales
        self.image_preprocessor = image_preprocessor
        self.match_iou = match_iou
        self.ignore_iou = ignore_iou

    @property
    def anchors(self):
        return tf.identity(self._anchors)

    def encode(
        self,
        image: tf.Tensor,
        gt_bboxes: tf.Tensor,
        gt_cls: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encodes ground truth bounding boxes and respective classes to target tensor.
        Additionally, preprocessing is applied to the image, e.g. applying
        normalization to be suitable as ResNet input.

        TODO provide a bit more context here or in the
        class documentation, or reference a blog article etc.

        Args:
            image (tf.Tensor): image as tf.Tensor with channels last
            gt_bboxes (tf.Tensor): ground truth bounding boxes relative centroids with y
                first ([ymin, xmin, ymax, xmax])
            gt_cls (tf.Tensor): ground truth classes as (1, M) tensor

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: image and target tensor with shape (N, 5) where
                (N, 1..4) are regression targets and and (N, 5) is the enoded classes
                where '-2' are ignored targets and '-1' are negative targets
        """
        # expect bboxes to yx relative corner encoded [ymin, xmin, ymax, xmax]
        # expects image to be shaped to self.image_shape

        # preprocess image @todo
        image = self.image_preprocessor(image)  # placeholder

        # encoding is only necessary if gt is present
        # otherwise all predictions are flase positives
        if tf.shape(gt_bboxes)[0] > 0:
            # scale bboxes up absolute centroids
            gt_bboxes = swap_xy(gt_bboxes)
            gt_bboxes = get_abs_bboxes(gt_bboxes, self.image_shape)
            gt_bboxes = convert_to_centroids(gt_bboxes)

            matched_gt_idx, positive_mask, ignore_mask = self._match(gt_bboxes)

            # encoding should start 0 for tf.one_hot, classes start 1, subtract 1
            gt_cls = gt_cls - 1

            # continue here
            # todo: try to make it faster by incorporating knowledge of postive_mask
            # combinging tf.concat([postitive_mask, matched_gt]) -> tf.map_fn
            # (N, 1) with M anchor boxes and second dim being matched object id
            matched_gt = tf.gather(gt_bboxes, matched_gt_idx)
            bboxes_target = self._encode_bboxes(matched_gt, self._anchors)
            gt_cls = tf.cast(gt_cls, tf.float32)
            cls_target = tf.gather(gt_cls, matched_gt_idx)

            # TODO maybe remove tf equal
            cls_target = tf.where(tf.equal(positive_mask, 1.0), cls_target, -1.0)
            cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
            cls_target = cls_target[..., tf.newaxis]
        else:
            bboxes_target = tf.zeros_like(self._anchors, tf.float32)
            num_anchors = tf.shape(self._anchors)[0]
            cls_target = -1.0 * tf.ones((num_anchors, 1), tf.float32)

        targets = tf.concat([bboxes_target, cls_target], axis=-1)

        return image, targets

    def _match(self, gt_bboxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Matches ground truth bounding boxes to anchors in self._anchors.

        Args:
            gt_bboxes (tf.Tensor): Ground truth bounding boxes as absolute centroids
                ([x, y, w, h])

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: tensors of shape (N,) containing
                ids of matched ground truth objects, positve matches, ignored
                predictions

        """

        # expected ground truth to be centroids
        # returns IOU for N `anchor_boxes` and M `gt_boxes` (N, M)
        iou_matrix = compute_iou(self._anchors, gt_bboxes)
        # todo argmax etc.
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        # get max index for each anchor box (N, )
        # every value is in the range 0..M-1
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        # get positive matches
        positive_mask = tf.greater_equal(max_iou, self.match_iou)

        # check if ignore values are supposed to be inclueded
        if self.ignore_iou is not None:
            # get ignore values
            negative_mask = tf.less(max_iou, self.ignore_iou)
            ignore_mask = tf.logical_not(tf.logical_or(negative_mask, positive_mask))
        else:
            negative_mask = tf.logical_not(positive_mask)
            ignore_mask = tf.zeros_like(positive_mask)

        negative_mask = tf.cast(negative_mask, tf.float32)
        ignore_mask = tf.cast(ignore_mask, tf.float32)
        positive_mask = tf.cast(positive_mask, tf.float32)

        return (
            matched_gt_idx,
            positive_mask,
            ignore_mask,
        )

    def _encode_bboxes(self, gt_bboxes: tf.Tensor, anchors: tf.Tensor) -> tf.Tensor:
        """Encodes anchor boxes based on Faster-RCNN approch.
        Better known as 'Faster-RCNN-box-coder'. Here is a really well written
        blog article describing it:
        https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/

        Using epsilon to avoid dividing by 0 is taken from:
        https://github.com/google/automl/blob/f2b4480703278250fb05abe38a2f4ecbb16ba463/efficientdet/object_detection/faster_rcnn_box_coder.py#L38

        Args:
            gt_bboxes (tf.Tensor): Ground truth anchor boxes encoded as absolute
                centroids [x, y, w, h]

        Returns:
            tf.Tensor: Encoded ground truth bounding boxes
        """
        # convert bboxes to

        # avoid division by 0
        w_gt = tf.maximum(gt_bboxes[:, 2], EPSILON)
        h_gt = tf.maximum(gt_bboxes[:, 3], EPSILON)
        x_center_gt, y_center_gt = gt_bboxes[:, 0], gt_bboxes[:, 1]

        x_center_anchor, y_center_anchor = anchors[:, 0], anchors[:, 1]
        w_anchor, h_anchor = anchors[:, 2], anchors[:, 3]

        target_x = (x_center_gt - x_center_anchor) / w_anchor
        target_y = (y_center_gt - y_center_anchor) / h_anchor
        target_w = tf.math.log(w_gt / w_anchor)
        target_h = tf.math.log(h_gt / h_anchor)

        if self.box_scales is not None:
            target_x = target_x * self.box_scales[0]
            target_y = target_y * self.box_scales[1]
            target_w = target_w * self.box_scales[2]
            target_h = target_h * self.box_scales[3]

        gt_bboxes_encoded = tf.stack(
            [target_x, target_y, target_w, target_h],
            axis=-1,
        )

        return gt_bboxes_encoded
