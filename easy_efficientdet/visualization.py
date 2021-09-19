import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Rectangle

from easy_efficientdet.utils import (
    convert_to_corners,
    get_abs_bboxes,
    setup_default_logger,
    swap_xy,
)

logger = setup_default_logger("visualization")

BBOX_FORMATS = frozenset(("abs_corner_xy", "abs_corner_yx", "rel_corner_xy",
                          "rel_corner_yx", "abs_center", "rel_center"))

# TODO MOVE this to separate file, check licensing issues
_default_colors = [
    "Aqua", "LimeGreen", "Magenta", "Chocolate", "AliceBlue", "Chartreuse", "Aqua",
    "Aquamarine", "Azure", "Beige", "Bisque", "BlanchedAlmond", "BlueViolet",
    "BurlyWood", "CadetBlue", "AntiqueWhite", "Coral", "CornflowerBlue", "Cornsilk",
    "Crimson", "Cyan", "DarkCyan", "DarkGoldenRod", "DarkGrey", "DarkKhaki",
    "DarkOrange", "DarkOrchid", "DarkSalmon", "DarkSeaGreen", "DarkTurquoise",
    "DarkViolet", "DeepPink", "DeepSkyBlue", "DodgerBlue", "FireBrick", "FloralWhite",
    "ForestGreen", "Fuchsia", "Gainsboro", "GhostWhite", "Gold", "GoldenRod", "Salmon",
    "Tan", "HoneyDew", "HotPink", "IndianRed", "Ivory", "Khaki", "Lavender",
    "LavenderBlush", "LawnGreen", "LemonChiffon", "LightBlue", "LightCoral",
    "LightCyan", "LightGoldenRodYellow", "LightGray", "LightGrey", "LightGreen",
    "LightPink", "LightSalmon", "LightSeaGreen", "LightSkyBlue", "LightSlateGray",
    "LightSlateGrey", "LightSteelBlue", "LightYellow", "Lime", "Linen",
    "MediumAquaMarine", "MediumOrchid", "MediumPurple", "MediumSeaGreen",
    "MediumSlateBlue", "MediumSpringGreen", "MediumTurquoise", "MediumVioletRed",
    "MintCream", "MistyRose", "Moccasin", "NavajoWhite", "OldLace", "Olive",
    "OliveDrab", "Orange", "OrangeRed", "Orchid", "PaleGoldenRod", "PaleGreen",
    "PaleTurquoise", "PaleVioletRed", "PapayaWhip", "PeachPuff", "Peru", "Pink", "Plum",
    "PowderBlue", "Purple", "Red", "RosyBrown", "RoyalBlue", "SaddleBrown", "Green",
    "SandyBrown", "SeaGreen", "SeaShell", "Sienna", "Silver", "SkyBlue", "SlateBlue",
    "SlateGray", "SlateGrey", "Snow", "SpringGreen", "SteelBlue", "GreenYellow", "Teal",
    "Thistle", "Tomato", "Turquoise", "Violet", "Wheat", "White", "WhiteSmoke",
    "Yellow", "YellowGreen"
]

_default_cmap = [mplc.get_named_colors_mapping()[c.lower()] for c in _default_colors]

_light_cmap = [mplc.to_rgba(c) for c in _default_cmap]
_light_cmap = [x[:3] + (x[3] * 0.4, ) for x in _light_cmap]


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


def plot_image_bbox(
    image,
    bboxes=None,
    bbox_cls_ids=None,
    bbox_format="abs_corner_xy",
    pred_bboxes=None,
    pred_bbox_cls_ids=None,
    pred_bbox_format="abs_center",
):
    """
    One plotting function to rule them all.

    Self-written, no licensing issues here.
    """

    # if pred_bboxes:
    #     raise NotImplementedError("prediction bboxes are not implemented, yet!")
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

    return ax
