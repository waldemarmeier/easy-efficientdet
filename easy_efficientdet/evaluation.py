from __future__ import annotations

import dataclasses
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

# isort: off
from easy_efficientdet._third_party.tf_object_detection_api.coco_evaluation \
    import CocoDetectionEvaluator
from easy_efficientdet._third_party.tf_object_detection_api.object_detection_evaluation\
    import PascalDetectionEvaluator
from easy_efficientdet.data.preprocessing import parse_od_record
from easy_efficientdet.utils import get_abs_bboxes, setup_default_logger, swap_xy
# isort: on

logger = setup_default_logger("evaluation")


def evaluate_od(dataset: tf.data.Dataset,
                prediction_model: tf.keras.Model,
                image_shape: Tuple[int],
                categories: List[Dict],
                include_metrics_per_category: bool = False,
                all_metrics_per_category: bool = False,
                is_data_parsed: bool = False,
                remove_sm_bbox_value: float = 0,
                subtract_one_from_cls: bool = True,
                batch_size: int = 1,
                log_frequency: int = 250):

    sample_counter = 0

    # prepare data
    if not is_data_parsed:
        dataset = dataset.map(parse_od_record)

    # parse data for coco evaluation
    dataset = dataset.map(partial(parse_data_cocoeval, image_shape=image_shape),
                          tf.data.experimental.AUTOTUNE)

    if remove_sm_bbox_value:
        dataset = dataset.map(
            partial(remove_small_gt, bbox_threshold=remove_sm_bbox_value),
            tf.data.experimental.AUTOTUNE)

    # create evaluator object
    evaluator = CocoDetectionEvaluator(
        categories=categories,
        include_metrics_per_category=include_metrics_per_category,
        all_metrics_per_category=all_metrics_per_category)

    # iterator though dataset to get groundtruth and respective predictions

    batched_data = dataset.padded_batch(batch_size,
                                        drop_remainder=False,
                                        padding_values={
                                            'image_id': b'',
                                            'image': .0,
                                            'bboxes': .0,
                                            'bbox_cls': tf.constant(-1, tf.int64),
                                            'box_cls_names': b''
                                        })
    batched_data = batched_data.prefetch(tf.data.experimental.AUTOTUNE)

    for sample_batch in batched_data:
        image_ids = sample_batch["image_id"].numpy().astype('S')

        curr_batch_size = tf.shape(sample_batch["image_id"])[0]

        for i in range(curr_batch_size):

            sample_mask = (sample_batch["bbox_cls"][i] >= 0)
            target_bboxes = sample_batch["bboxes"][i]
            target_bboxes = tf.boolean_mask(target_bboxes, sample_mask)
            target_bboxes = swap_xy(target_bboxes).numpy()

            target_cls = sample_batch["bbox_cls"][i]
            target_cls = tf.boolean_mask(target_cls, sample_mask)
            target_cls = target_cls.numpy()

            gt = GroundTruth(groundtruth_boxes=target_bboxes,
                             groundtruth_classes=target_cls)

            if subtract_one_from_cls:
                gt = gt.subtract_one_from_cls()

            gt_dict = dataclasses.asdict(gt)
            # print('gt', gt_dict)
            image_id = image_ids[i]
            evaluator.add_single_ground_truth_image_info(image_id=image_id,
                                                         groundtruth_dict=gt_dict)

        logger.debug("evaluating image with id: {}".format(image_id))

        images = sample_batch["image"]

        # inference, add results ass detections to evaluator
        predictions = prediction_model(images)

        for i in range(curr_batch_size):

            parsed_prediction = parse_combined_nms_obj(predictions, i)

            # swap x,y
            # parsed_prediction.swap_box_xy()
            prediction_dict = dataclasses.asdict(parsed_prediction)
            # print("pred", prediction_dict)
            image_id = image_ids[i]
            evaluator.add_single_detected_image_info(image_id=image_id,
                                                     detections_dict=prediction_dict)

            sample_counter += 1

            if (log_frequency > 0) and ((sample_counter % log_frequency) == 0):
                logger.info(f"currently at sample {sample_counter}")

    eval_results = evaluator.evaluate()
    return eval_results


def evaluate_od_pascal(dataset: tf.data.Dataset,
                       prediction_model: tf.keras.Model,
                       image_shape: Tuple[int],
                       categories: List[Dict],
                       include_metrics_per_category: bool = False,
                       all_metrics_per_category: bool = False,
                       is_data_parsed: bool = False,
                       remove_sm_bbox_value: float = 0,
                       subtract_one_from_cls: bool = True,
                       batch_size: int = 1):

    # prepare data
    if not is_data_parsed:
        dataset = dataset.map(parse_od_record)

    # parse data for coco evaluation
    dataset = dataset.map(partial(parse_data_cocoeval, image_shape=image_shape),
                          tf.data.experimental.AUTOTUNE)

    if remove_sm_bbox_value:
        dataset = dataset.map(
            partial(remove_small_gt, bbox_threshold=remove_sm_bbox_value),
            tf.data.experimental.AUTOTUNE)

    # create evaluator object
    evaluator = PascalDetectionEvaluator(categories,
                                         matching_iou_threshold=0.5,
                                         nms_iou_threshold=1.0,
                                         nms_max_output_boxes=100)

    batched_data = dataset.padded_batch(batch_size,
                                        drop_remainder=False,
                                        padding_values={
                                            'image_id': b'',
                                            'image': .0,
                                            'bboxes': .0,
                                            'bbox_cls': tf.constant(-1, tf.int64),
                                            'box_cls_names': b''
                                        })
    batched_data = batched_data.prefetch(tf.data.experimental.AUTOTUNE)

    for sample_batch in batched_data:
        image_ids = sample_batch["image_id"].numpy().astype('S')

        curr_batch_size = tf.shape(sample_batch["image_id"])[0]

        for i in range(curr_batch_size):

            sample_mask = (sample_batch["bbox_cls"][i] >= 0)
            target_bboxes = sample_batch["bboxes"][i]
            target_bboxes = tf.boolean_mask(target_bboxes, sample_mask)
            target_bboxes = swap_xy(target_bboxes).numpy()

            target_cls = sample_batch["bbox_cls"][i]
            target_cls = tf.boolean_mask(target_cls, sample_mask)
            target_cls = target_cls.numpy()

            gt = GroundTruth(groundtruth_boxes=target_bboxes,
                             groundtruth_classes=target_cls)

            gt_dict = dataclasses.asdict(gt)

            image_id = image_ids[i]
            evaluator.add_single_ground_truth_image_info(image_id=image_id,
                                                         groundtruth_dict=gt_dict)

        logger.debug("evaluating image with id: {}".format(image_id))

        images = sample_batch["image"]

        # inference, add results ass detections to evaluator
        predictions = prediction_model(images)

        for i in range(curr_batch_size):

            parsed_prediction = parse_combined_nms_obj(predictions, i)
            parsed_prediction = parsed_prediction.add_one_to_cls()
            # swap x,y
            # parsed_prediction.swap_box_xy()
            prediction_dict = dataclasses.asdict(parsed_prediction)
            # print("pred", prediction_dict)
            image_id = image_ids[i]
            evaluator.add_single_detected_image_info(image_id=image_id,
                                                     detections_dict=prediction_dict)

    eval_results = evaluator.evaluate()
    return eval_results


def parse_data_cocoeval(sample, image_shape):

    image = sample["image"]
    #    image = tf.image.grayscale_to_rgb(image)
    image = tf.image.resize(image, image_shape)
    image_id = sample["filename"]  # sample["source_id"][0]
    bboxes = sample["bboxes"]
    bboxes = get_abs_bboxes(bboxes, image_shape)
    bbox_cls = sample["bbox_cls"]
    bbox_cls_names = sample["bbox_cls_names"]

    return {
        "image_id": image_id,
        "image": image,
        "bboxes": bboxes,
        "bbox_cls": bbox_cls,
        "box_cls_names": bbox_cls_names
    }


# test if this really works
def remove_small_gt(sample, bbox_threshold=5):
    bboxes = sample["bboxes"]
    bbox_cls = sample["bbox_cls"]
    bbox_cls_names = sample["box_cls_names"]

    w = bboxes[:, 2] - bboxes[:, 0]
    h = bboxes[:, 3] - bboxes[:, 1]

    keep_bboxes = tf.math.logical_and(tf.greater(w, bbox_threshold),
                                      tf.greater(h, bbox_threshold),
                                      name="check_bbox_width_and_height")

    bboxes = tf.boolean_mask(bboxes, keep_bboxes, axis=0)
    bbox_cls = tf.boolean_mask(bbox_cls, keep_bboxes, axis=0)
    bbox_cls_names = tf.boolean_mask(bbox_cls_names, keep_bboxes, axis=0)

    return {
        "image_id": sample["image_id"],
        "image": sample["image"],
        "bboxes": bboxes,
        "bbox_cls": bbox_cls,
        "box_cls_names": bbox_cls_names
    }


@dataclasses.dataclass
class GroundTruth:
    groundtruth_boxes: np.ndarray
    groundtruth_classes: np.ndarray
    groundtruth_is_crowd: Optional[bool] = None
    groundtruth_area: Optional[Any] = None
    groundtruth_keypoints: Optional[Any] = None
    groundtruth_keypoint_visibilities: Optional[Any] = None

    def subtract_one_from_cls(self) -> GroundTruth:
        return self.__class__(
            groundtruth_boxes=self.groundtruth_boxes,
            groundtruth_classes=self.groundtruth_classes - 1,
            groundtruth_is_crowd=self.groundtruth_is_crowd,
            groundtruth_area=self.groundtruth_area,
            groundtruth_keypoints=self.groundtruth_keypoints,
            groundtruth_keypoint_visibilities=self.groundtruth_keypoint_visibilities,
        )


@dataclasses.dataclass
class DetectionResult:
    detection_boxes: np.ndarray
    detection_scores: np.ndarray
    detection_classes: np.ndarray
    detection_keypoints: Optional[np.ndarray] = None

    def swap_box_xy(self) -> None:
        self.detection_boxes = swap_xy(self.detection_boxes).numpy()

    def add_one_to_cls(self) -> DetectionResult:
        return self.__class__(detection_boxes=self.detection_boxes,
                              detection_scores=self.detection_scores,
                              detection_classes=self.detection_classes + 1,
                              detection_keypoints=self.detection_keypoints)


def parse_combined_nms_obj(cnms_objs, i=0) -> DetectionResult:

    cnms_objs = list(map(lambda x: x.numpy(), cnms_objs))
    nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = cnms_objs

    num_detections = valid_detections[i]
    boxes = nmsed_boxes[i][:num_detections, :]
    scores = nmsed_scores[i, :num_detections]

    labels = nmsed_classes[i, :num_detections]
    labels = labels.astype(np.int32)

    return DetectionResult(detection_boxes=boxes,
                           detection_scores=scores,
                           detection_classes=labels)
