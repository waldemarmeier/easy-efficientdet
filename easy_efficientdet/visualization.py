import random
from typing import Dict, List, Optional, Sequence

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.colors import cnames
from matplotlib.patches import Rectangle

from easy_efficientdet.utils import (
    LabelMapType,
    convert_to_corners,
    get_abs_bboxes,
    setup_default_logger,
    swap_xy,
    tf_round,
)

logger = setup_default_logger("visualization")

# TODO convert this to an enum
BBOX_FORMATS = frozenset(("abs_corner_xy", "abs_corner_yx", "rel_corner_xy",
                          "rel_corner_yx", "abs_center", "rel_center"))

PREDICTION_COLOR_ALPHA = .4
_default_colors = list(cnames.keys())
random.Random(123).shuffle(_default_colors)

_default_cmap = [mplc.get_named_colors_mapping()[c.lower()] for c in _default_colors]

_light_cmap = [mplc.to_rgba(c) for c in _default_cmap]
_light_cmap = [x[:3] + (x[3] * PREDICTION_COLOR_ALPHA, ) for x in _light_cmap]


def _prepare_bboxes_for_plt(bboxes, bbox_format, image_shape):

    if len(bboxes.shape) > 2:
        bboxes = bboxes[0]
        logger.warning("Provided batch of bounding boxes. Using first one.")

    if "center" in bbox_format:
        if "rel" in bbox_format:
            raise NotImplementedError(f"bbox format not implemented {bbox_format}")

        bboxes = convert_to_corners(bboxes)

    if "yx" in bbox_format:
        bboxes = swap_xy(bboxes)

    if "rel" in bbox_format:
        bboxes = get_abs_bboxes(bboxes, image_shape)

    return bboxes


def generate_prediction_labels(
    pred_bbox_cls_ids: Optional[Sequence[int]] = None,
    pred_bboxes_probs: Optional[List[float]] = None,
    label_map: Optional[LabelMapType] = None,
) -> Sequence[str]:
    # nothing provided
    if pred_bbox_cls_ids is None and pred_bboxes_probs is None:
        raise ValueError("class ids and probabilities are empty")

    if pred_bbox_cls_ids is not None and pred_bboxes_probs is not None:
        if len(pred_bbox_cls_ids) != len(pred_bboxes_probs):
            raise ValueError("Number of predictions and respective probabilities does"
                             f" not match: {pred_bbox_cls_ids} vs. {pred_bboxes_probs}")

    id_name_map: Optional[Dict[int, str]] = None
    if label_map is not None:
        id_name_map = {el['id']: el['name'] for el in label_map}

    probs = None
    if pred_bboxes_probs is not None:
        probs = list(map(lambda x: str(tf_round(x, 2).numpy()) + "%",
                         pred_bboxes_probs))

    names_or_ids = None
    if pred_bbox_cls_ids is not None:
        if id_name_map is not None:
            names_or_ids = list(map(lambda x: f"{id_name_map[x]}", pred_bbox_cls_ids))
        else:
            names_or_ids = list(map(lambda x: f"id {x}", pred_bbox_cls_ids))

    if probs is not None and names_or_ids is not None:
        return [f"{name}: {p}" for name, p in zip(names_or_ids, probs)]
    if probs is not None:
        return probs
    else:
        return names_or_ids


def plot_image_bbox(image,
                    bboxes=None,
                    bbox_cls_ids=None,
                    bbox_format="rel_corner_yx",
                    pred_bboxes=None,
                    pred_bbox_cls_ids: Optional[Sequence[int]] = None,
                    pred_bbox_format="rel_corner_yx",
                    pred_bboxes_probs: Optional[Sequence[float]] = None,
                    label_map: Optional[LabelMapType] = None,
                    figsize: Optional[Sequence[int]] = None,
                    fontsize: int = 12):
    """
    One plotting function to rule them all.
    """

    if figsize is not None:
        plt.figure(figsize=figsize)

    if bboxes is not None:
        if bbox_format not in BBOX_FORMATS:
            raise ValueError(
                "Wrong value for bbox_format: '{bbox_format}'. Allowed values: {values}"
                .format(bbox_format=bbox_format, values=", ".join(BBOX_FORMATS)))

    if pred_bboxes is not None:
        if pred_bbox_format not in BBOX_FORMATS:
            raise ValueError(
                "Wrong value for pred_bbox_format: '{bbox_format}'. Allowed values: "
                "{values}".format(bbox_format=pred_bbox_format,
                                  values=", ".join(BBOX_FORMATS)))

    image_shape = image.shape

    if len(image.shape) > 3:
        image = image[0]
        logger.warning(
            "Provided batch of images. Using first image and bboxes for plotting.")

    if "float" in str(image.dtype):
        if tf.reduce_max(tf.reshape(image, (-1, ))) > 1.0:
            image = image / 255.0

    if image_shape[-1] == 1:
        image = tf.tile(image, (1, 1, 3))

    plt.imshow(image)
    ax = plt.gca()

    if bboxes is not None:

        bboxes = _prepare_bboxes_for_plt(
            bboxes=bboxes,
            bbox_format=bbox_format,
            image_shape=image_shape,
        )

        if bbox_cls_ids is None:
            for bbox in bboxes:
                ax.add_patch(
                    Rectangle(
                        bbox[:2],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        fill=False,
                        edgecolor="blue",
                    ))
        else:
            if bbox_cls_ids.shape[0] != bboxes.shape[0]:
                raise ValueError(
                    "Unequal number of ground truth bounding boxes and respective "
                    "labels provided")

            for bbox, bbox_cls_id in zip(bboxes, bbox_cls_ids):
                ax.add_patch(
                    Rectangle(bbox[:2],
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1],
                              linewidth=1.0,
                              fill=False,
                              edgecolor=_light_cmap[bbox_cls_id]))

    if pred_bboxes is not None:
        pred_bboxes = _prepare_bboxes_for_plt(
            bboxes=pred_bboxes,
            bbox_format=pred_bbox_format,
            image_shape=image_shape,
        )

        if pred_bbox_cls_ids is None:
            for bbox in pred_bboxes:
                ax.add_patch(
                    Rectangle(
                        bbox[:2],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        fill=False,
                        edgecolor="red",
                    ))
        else:
            if pred_bbox_cls_ids.shape[0] != pred_bboxes.shape[0]:
                raise ValueError(
                    "Unequal number of prediction bounding boxes and respective labels "
                    "provided")

            for bbox, bbox_cls_id in zip(pred_bboxes, pred_bbox_cls_ids):
                ax.add_patch(
                    Rectangle(
                        bbox[:2],
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1],
                        linewidth=1.0,
                        fill=False,
                        edgecolor=_default_cmap[bbox_cls_id],
                    ))

    if pred_bbox_cls_ids is not None and pred_bboxes is not None:

        label_texts = generate_prediction_labels(pred_bbox_cls_ids, pred_bboxes_probs,
                                                 label_map)

        for text, bbox, bbox_cls_id in zip(label_texts, pred_bboxes, pred_bbox_cls_ids):
            ax.text(bbox[0],
                    bbox[1],
                    text,
                    clip_box=ax.clipbox,
                    clip_on=True,
                    bbox={
                        "facecolor": _default_colors[bbox_cls_id],
                        "alpha": PREDICTION_COLOR_ALPHA
                    },
                    fontsize=fontsize)

    return ax
