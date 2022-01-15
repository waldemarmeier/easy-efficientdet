from functools import partial
from typing import Optional, Sequence

import tensorflow as tf

from easy_efficientdet._third_party.decoder import DecodePredictions  # noqa F401
from easy_efficientdet.boxencoding import generate_anchor_boxes
from easy_efficientdet.utils import convert_to_corners


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

    decoder = DecodePredictions(num_classes=num_cls,
                                image_shape=image_shape,
                                confidence_threshold=confidence_threshold,
                                nms_iou_threshold=nms_iou_threshold,
                                max_detections_per_class=max_detections_per_class,
                                max_detections=max_detections)

    if resize:
        inp = x = tf.keras.Input((None, None, image_shape[2]))
        x = tf.image.resize(x, image_shape[:2])
    else:
        inp = x = tf.keras.Input(image_shape)
    x = model(x, training=False)
    x = decoder(x)
    inference_model = tf.keras.Model(inp, x, name=f"inference_{model.name}")

    return inference_model


class DecodePredictionsSoft(tf.keras.layers.Layer):
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
    def __init__(self,
                 num_classes: int = 4,
                 image_shape: Sequence[int] = (512, 512),
                 confidence_threshold: float = 0.05,
                 nms_iou_threshold: float = 0.5,
                 max_detections_per_class: int = 100,
                 max_detections: int = 100,
                 box_variance: Optional[Sequence[float]] = None,
                 sigma: float = .05,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections
        self.sigma = sigma
        # TODO generate_anchor_boxes must be better configurable
        self._anchor_box = generate_anchor_boxes(image_shape)
        self.box_variance = box_variance
        self._soft_nms_fun = partial(tf.image.non_max_suppression_with_scores,
                                     max_output_size=max_detections_per_class,
                                     iou_threshold=nms_iou_threshold,
                                     score_threshold=confidence_threshold,
                                     soft_nms_sigma=sigma)

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
        #         image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        # .get_anchors(image_shape[1], image_shape[2])
        box_predictions = predictions[:, :, :4]
        cls_predictions = tf.nn.sigmoid(predictions[:, :, 4:])
        boxes = self._decode_box_predictions(self._anchor_box[None, ...],
                                             box_predictions)

        num_classes = tf.shape(cls_predictions)[-1]
        batch_size = tf.shape(predictions)[0]  # predictions.shape[0]
        cls_pred_nms = tf.TensorArray(dtype=tf.int64,
                                      size=batch_size,
                                      dynamic_size=True)
        scores_pred_nms = tf.TensorArray(dtype=tf.float32,
                                         size=batch_size,
                                         dynamic_size=True)
        box_pred_nms = tf.TensorArray(dtype=tf.float32,
                                      size=batch_size,
                                      dynamic_size=True)
        num_detections = tf.TensorArray(dtype=tf.int32,
                                        size=batch_size,
                                        dynamic_size=True)

        preds_all = tf.concat([boxes, cls_predictions], axis=-1)

        # TODO rename to batch_idx
        for batch_num in tf.range(batch_size):
            sample = preds_all[batch_num]

            max_cls_idx_per_box = tf.argmax(sample[:, 4:], output_type=tf.int32, axis=1)

            num_boxes = tf.shape(max_cls_idx_per_box)[0]
            select_score_per_box_idx = tf.concat(
                (tf.range(num_boxes)[..., tf.newaxis],
                 4 + max_cls_idx_per_box[..., tf.newaxis]),
                axis=-1)
            max_score_per_box = tf.gather_nd(sample, select_score_per_box_idx)

            keep_boxes = tf.greater_equal(max_score_per_box, self.confidence_threshold)
            sample = tf.boolean_mask(sample, keep_boxes, axis=0)
            max_cls_idx_per_box = tf.boolean_mask(max_cls_idx_per_box,
                                                  keep_boxes,
                                                  axis=0)

            cls_pred_nms_cls = tf.TensorArray(dtype=tf.int64,
                                              size=num_classes,
                                              dynamic_size=True)
            scores_pred_nms_cls = tf.TensorArray(dtype=tf.float32,
                                                 size=num_classes,
                                                 dynamic_size=True)
            box_pred_nms_cls = tf.TensorArray(dtype=tf.float32,
                                              size=num_classes,
                                              dynamic_size=True)
            num_detections_cls = tf.TensorArray(dtype=tf.int32,
                                                size=num_classes,
                                                dynamic_size=True)

            # TODO cls_nums -> cls_idx
            for cls_num in tf.range(num_classes):

                # reduce to preds for which cls_num has highest score for anchor box
                sample_cls = tf.boolean_mask(sample,
                                             tf.equal(max_cls_idx_per_box, cls_num),
                                             axis=0)

                idx_cls, scores_cls = self._soft_nms_fun(boxes=sample_cls[:, :4],
                                                         scores=sample_cls[:,
                                                                           cls_num + 4])
                num_valid_detections_cls = tf.shape(idx_cls)[0]
                num_detections_cls = num_detections_cls.write(
                    cls_num, num_valid_detections_cls)
                diff_detections_cls = \
                    self.max_detections_per_class - num_valid_detections_cls

                sub_sample_cls = tf.gather(sample_cls, idx_cls)
                box_pred_nms_sample_cls = tf.pad(sub_sample_cls[:, :4],
                                                 [[0, diff_detections_cls], [0, 0]],
                                                 "CONSTANT", 0,
                                                 "box_pred_nms_sample_cls")
                box_pred_nms_cls = box_pred_nms_cls.write(cls_num,
                                                          box_pred_nms_sample_cls)

                scores_sample_cls = tf.pad(scores_cls, [[0, diff_detections_cls]],
                                           "CONSTANT", 0, "scores_sample_cls")
                scores_pred_nms_cls = scores_pred_nms_cls.write(
                    cls_num, scores_sample_cls)

                cls_pred_nms_sample_cls = tf.math.argmax(sub_sample_cls[:, 4:], axis=-1)
                cls_pred_nms_sample_cls = tf.pad(cls_pred_nms_sample_cls,
                                                 [[0, diff_detections_cls]], "CONSTANT",
                                                 -1, "cls_pred_nms_sample_cls")
                cls_pred_nms_cls = cls_pred_nms_cls.write(cls_num,
                                                          cls_pred_nms_sample_cls)

            num_detections_all = tf.reduce_sum(num_detections_cls.concat())
            num_detections = num_detections.write(
                batch_num, tf.math.minimum(num_detections_all, self.max_detections))

            cls_pred_nms_cls = cls_pred_nms_cls.concat()
            pred_pos = tf.greater(cls_pred_nms_cls, -1)

            # write to batch level tensors
            #   - pad if less than max detections
            #   - drop detections with lowest scores if more than max detections
            if num_detections_all <= self.max_detections:
                # less than max_predictions -> pad tensors
                diff_detections = self.max_detections - num_detections_all

                box_pred_nms_cls = box_pred_nms_cls.concat()
                box_pred_nms_cls = tf.boolean_mask(box_pred_nms_cls, pred_pos, axis=0)
                box_pred_nms_cls = tf.pad(box_pred_nms_cls,
                                          [[0, diff_detections], [0, 0]], "CONSTANT",
                                          .0)
                box_pred_nms = box_pred_nms.write(batch_num, box_pred_nms_cls)

                scores_pred_nms_cls = scores_pred_nms_cls.concat()
                scores_pred_nms_cls = tf.boolean_mask(scores_pred_nms_cls,
                                                      pred_pos,
                                                      axis=0)
                scores_pred_nms_cls = tf.pad(scores_pred_nms_cls,
                                             [[0, diff_detections]], "CONSTANT", .0)
                scores_pred_nms = scores_pred_nms.write(batch_num, scores_pred_nms_cls)

                cls_pred_nms_cls = tf.boolean_mask(cls_pred_nms_cls, pred_pos, axis=0)
                cls_pred_nms_cls = tf.pad(cls_pred_nms_cls, [[0, diff_detections]],
                                          "CONSTANT", -1)
                cls_pred_nms = cls_pred_nms.write(batch_num, cls_pred_nms_cls)
            else:
                # tf.print("more than max detections")
                scores_pred_nms_cls = scores_pred_nms_cls.concat()
                scores_pred_nms_cls = tf.boolean_mask(scores_pred_nms_cls,
                                                      pred_pos,
                                                      axis=0)
                # take self.max_detection of best scores positions
                best_scores_idx = tf.argsort(scores_pred_nms_cls,
                                             axis=-1,
                                             direction='DESCENDING',
                                             stable=True)[:self.max_detections]

                # get best score and write them to batch level tensor arr
                scores_pred_nms_cls = tf.gather(scores_pred_nms_cls, best_scores_idx)
                scores_pred_nms = scores_pred_nms.write(batch_num, scores_pred_nms_cls)

                # get for best scores the respective bboxes and write them to
                # batch level tensor arr
                box_pred_nms_cls = box_pred_nms_cls.concat()
                box_pred_nms_cls = tf.boolean_mask(box_pred_nms_cls, pred_pos, axis=0)
                box_pred_nms_cls = tf.gather(box_pred_nms_cls, best_scores_idx)
                box_pred_nms = box_pred_nms.write(batch_num, box_pred_nms_cls)

                # get for best scores the respective classes and write them
                # to batch level tensor
                cls_pred_nms_cls = tf.boolean_mask(cls_pred_nms_cls, pred_pos, axis=0)
                cls_pred_nms_cls = tf.gather(cls_pred_nms_cls, best_scores_idx)
                cls_pred_nms = cls_pred_nms.write(batch_num, cls_pred_nms_cls)

        # nms_results = AdapterNMSResult(valid_detections = num_detections.stack(),
        #                                 nmsed_boxes = box_pred_nms.stack(),
        #                                 nmsed_scores = scores_pred_nms.stack(),
        #                                 nmsed_classes = cls_pred_nms.stack())

        return {
            'valid_detections': num_detections.stack(),
            'nmsed_boxes': box_pred_nms.stack(),
            'nmsed_scores': scores_pred_nms.stack(),
            "nmsed_classes": cls_pred_nms.stack()
        }
