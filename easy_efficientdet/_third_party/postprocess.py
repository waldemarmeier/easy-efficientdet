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
# ==============================================================================

import tensorflow as tf

# TFLite-specific constants.
TFLITE_MAX_CLASSES_PER_DETECTION = 1
TFLITE_DETECTION_POSTPROCESS_FUNC = 'TFLite_Detection_PostProcess'
# TFLite fast NMS == postprocess_global (less accurate)
# TFLite regular NMS == postprocess_per_class
TFLITE_USE_REGULAR_NMS = False


def tflite_nms_implements_signature(num_classes: int,
                                    iou_thresh: float = .5,
                                    score_thresh: float = float("-inf"),
                                    max_detections=100):
    """`experimental_implements` signature for TFLite's custom NMS op.
    This signature encodes the arguments to correctly initialize TFLite's custom
    post-processing op in the MLIR converter.
    For details on `experimental_implements` see here:
    https://www.tensorflow.org/api_docs/python/tf/function
    Args:
    params: a dict of parameters.
    Returns:
    String encoding of a map from attribute keys to values.
    """
    scale_value = 1.0
    #     nms_configs = params['nms_configs']
    #     iou_thresh = nms_configs['iou_thresh'] or 0.5
    #     score_thresh = nms_configs['score_thresh'] or float('-inf')
    #     max_detections = params['tflite_max_detections']

    implements_signature = [
        'name: "%s"' % TFLITE_DETECTION_POSTPROCESS_FUNC,
        'attr { key: "max_detections" value { i: %d } }' % max_detections,
        'attr { key: "max_classes_per_detection" value { i: %d } }' %
        TFLITE_MAX_CLASSES_PER_DETECTION,
        'attr { key: "use_regular_nms" value { b: %s } }' %
        str(TFLITE_USE_REGULAR_NMS).lower(),
        'attr { key: "nms_score_threshold" value { f: %f } }' % score_thresh,
        'attr { key: "nms_iou_threshold" value { f: %f } }' % iou_thresh,
        'attr { key: "y_scale" value { f: %f } }' % scale_value,
        'attr { key: "x_scale" value { f: %f } }' % scale_value,
        'attr { key: "h_scale" value { f: %f } }' % scale_value,
        'attr { key: "w_scale" value { f: %f } }' % scale_value,
        'attr { key: "num_classes" value { i: %d } }' % num_classes
    ]
    implements_signature = ' '.join(implements_signature)
    return implements_signature


def postprocess_tflite(num_classes: int, iou_thresh: float, score_thresh: float,
                       max_detections: int, cls_outputs: tf.Tensor,
                       box_outputs: tf.Tensor, decoded_anchors: tf.Tensor):
    """Post processing for conversion to TFLite.
    Mathematically same as postprocess_global, except that the last portion of the
    TF graph constitutes a dummy `tf.function` that contains an annotation for
    conversion to TFLite's custom NMS op. Using this custom op allows features
    like post-training quantization & accelerator support.
    NOTE: This function does NOT return a valid output, and is only meant to
    generate a SavedModel for TFLite conversion via MLIR.
    For TFLite op details, see tensorflow/lite/kernels/detection_postprocess.cc
    Args:
    params: a dict of parameters.
    cls_outputs: a list of tensors for classes, each tensor denotes a level of
      logits with shape [1, H, W, num_class * num_anchors].
    box_outputs: a list of tensors for boxes, each tensor ddenotes a level of
      boxes with shape [1, H, W, 4 * num_anchors]. Each box format is [y_min,
      x_min, y_max, x_man].
    Returns:
    A (dummy) tuple of (boxes, scores, classess, valid_len).
    """
    scores = cls_outputs

    #   box_outputs, scores, decoded_anchors = tflite_pre_nms(params, cls_outputs,
    #                                                         box_outputs)

    # There is no TF equivalent for TFLite's custom post-processing op.
    # So we add an 'empty' composite function here, that is legalized to the
    # custom op with MLIR.
    # For details, see:
    # tensorflow/compiler/mlir/lite/utils/nms_utils.cc
    @tf.function(experimental_implements=tflite_nms_implements_signature(
        num_classes, iou_thresh, score_thresh, max_detections))
    # pylint: disable=g-unused-argument,unused-argument
    def dummy_post_processing(box_encodings, class_predictions, anchor_boxes):
        boxes = tf.constant(0.0, dtype=tf.float32, name='boxes')
        scores = tf.constant(0.0, dtype=tf.float32, name='scores')
        classes = tf.constant(0.0, dtype=tf.float32, name='classes')
        num_detections = tf.constant(0.0, dtype=tf.float32, name='num_detections')
        return boxes, classes, scores, num_detections

    return dummy_post_processing(box_outputs, scores, decoded_anchors)[::-1]
