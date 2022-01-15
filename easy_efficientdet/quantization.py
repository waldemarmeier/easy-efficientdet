import tempfile
from datetime import datetime
from typing import Generator, Optional, Sequence, Set

import tensorflow as tf

from easy_efficientdet._third_party.postprocess import postprocess_tflite
from easy_efficientdet.utils import setup_default_logger

logger = setup_default_logger("quantization")


class _OptimzationType:

    INT8 = "int8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"

    def __init__(self, ):
        self.types = self.valid_types()

    def __contains__(self, other) -> bool:
        return other in self.types

    @classmethod
    def valid_types(cls, ) -> Set[str]:
        return {cls.INT8, cls.FLOAT16, cls.FLOAT32}


OptimzationType = _OptimzationType()


class ExportModel(tf.Module):
    """Model to be exported as SavedModel/TFLite format."""
    def __init__(self, num_cls: int, iou_thresh: float, score_thresh: float,
                 max_detections: int, model: tf.keras.Model, anchors: tf.Tensor):
        super().__init__()
        self.num_cls = num_cls
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh
        self.max_detection = max_detections
        self.model = model
        self.anchors = anchors  # normalized

    @tf.function
    def __call__(self, imgs):
        x = self.model(imgs, training=False)
        cls_output = tf.math.sigmoid(x[..., 4:])
        box_output = x[..., :4]
        return postprocess_tflite(num_classes=self.num_cls,
                                  cls_outputs=cls_output,
                                  box_outputs=box_output,
                                  decoded_anchors=self.anchors,
                                  iou_thresh=self.score_thresh,
                                  score_thresh=self.score_thresh,
                                  max_detections=self.max_detection)


def quantize(export_model: tf.Module,
             opt_type: OptimzationType,
             image_shape: Sequence[int],
             representative_dataset: Generator[tf.Tensor, None, None] = None,
             filename: Optional[str] = None) -> bytes:

    if opt_type not in OptimzationType:
        raise ValueError(f"Not valid optimatization type {opt_type}. "
                         "Optimization type must be in "
                         f"{OptimzationType.valid_types()}")

    if (opt_type == OptimzationType.INT8) and (representative_dataset is None):
        raise ValueError("For INT8 quantization type a respresentative dataset "
                         "has to be provided")
    tmp_prefix = "opt_" + opt_type
    tmp_suffix = datetime.now().strftime("%Y%m%d%H%M%S")

    with tempfile.TemporaryDirectory(prefix=tmp_prefix, suffix=tmp_suffix) as tmpf:

        input_spec = tf.TensorSpec(shape=[1, *image_shape],
                                   dtype=tf.float32,
                                   name='images')
        signatures = export_model.__call__.get_concrete_function(input_spec)
        tf.saved_model.save(export_model, tmpf, signatures=signatures)

        converter = tf.lite.TFLiteConverter.from_saved_model(tmpf)

        if opt_type == OptimzationType.FLOAT32:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float32]
        elif opt_type == OptimzationType.FLOAT16:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif opt_type == OptimzationType.INT8:
            converter.experimental_new_quantizer = True
            converter.representative_dataset = representative_dataset
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.inference_input_type = tf.uint8
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS
            ]

        tflite_model = converter.convert()

        if filename is not None:
            with open(filename, 'wb') as fp:
                fp.write(tflite_model)

        return tflite_model
