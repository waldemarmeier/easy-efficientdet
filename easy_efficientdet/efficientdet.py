import re
from numbers import Number
from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf

from .layers import BiFPN, BoxPredLayer, ClassPredLayer, PreBiFPN
from .utils import setup_default_logger

# honestly, should rename this file to 'model.py'
_LAYERS_PATTERN = re.compile("(block[12356][a-z])")
# _name_getter = attrgetter("name")

logger = setup_default_logger("efficientdet-model")

# params for efficientdet
# 7x is at index 8
NUM_W_BiFPN = (64, 88, 112, 160, 224, 288, 384)
IMAGE_RESOLUTION = (512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536)
BIFPN_LAYERS = (3, 4, 5, 6, 7, 7, 8, 8, 8)
NUM_HEAD_LAYERS = (3, 3, 3, 4, 4, 4, 5, 5, 5)
BACKBONE_NUM = (0, 1, 2, 3, 4, 5, 6, 6, 7)

# fusion parameters, currently not used
FAST_FUSION_EPSILON = 1e-4
WEIGHTED_FUSION_TYPE = "fast_attention"  # currently no other type implemented

# Batch normalization parameters
BN_MOMENTUM = 0.99
BN_EPSILON = 0.001


def EfficientDet(
    version: int = 0,
    num_cls: int = 4,
    num_anchors: int = 9,
    image_size: Optional[Union[Tuple, int]] = None,
):
    image_shape = _get_image_size(image_size, version)
    backbone_num = BACKBONE_NUM[version]
    num_w_bifpn = NUM_W_BiFPN[version]
    num_head_layers = NUM_HEAD_LAYERS[version]
    num_bifpn_layers = BIFPN_LAYERS[version]

    return effdet_functional(version=version,
                             num_cls=num_cls,
                             num_anchors=num_anchors,
                             image_shape=image_shape,
                             backbone_num=backbone_num,
                             bn_momentum=BN_MOMENTUM,
                             bn_epsilon=BN_EPSILON,
                             num_w_bifpn=num_w_bifpn,
                             num_bifpn_layers=num_bifpn_layers,
                             num_head_layers=num_head_layers,
                             fast_fusion_epsilon=FAST_FUSION_EPSILON,
                             weighted_fusion_type=WEIGHTED_FUSION_TYPE)


def effdet_functional(version: int, num_cls: int, num_anchors: int,
                      image_shape: Sequence[int], backbone_num: int, bn_momentum: float,
                      bn_epsilon: float, num_w_bifpn: int, num_bifpn_layers: int,
                      num_head_layers: int, fast_fusion_epsilon: float,
                      weighted_fusion_type: float):

    efficientnet = _create_backbone(
        backbone_num,
        image_shape,
    )
    fpn_input_layers = _extract_layers(efficientnet)
    # just take the last 3
    fpn_input_layers = fpn_input_layers[2:]

    backbone = tf.keras.Model(inputs=efficientnet.input,
                              outputs=fpn_input_layers,
                              name="efficientnet_backbone")

    pre_bifpn = PreBiFPN(
        num_w_bifpn,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
    )
    bifpn = BiFPN(
        num_bifpn_layers=num_bifpn_layers,
        num_w_bifpn=num_w_bifpn,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
    )
    # layer for box predictions
    box_pred = BoxPredLayer(
        width=num_w_bifpn,
        depth=num_head_layers,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        num_anchors=num_anchors,
    )

    cls_pred = ClassPredLayer(
        width=num_w_bifpn,
        depth=num_head_layers,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        num_anchors=num_anchors,
        num_cls=num_cls,
    )

    inp = tf.keras.layers.Input(image_shape)
    x = inp

    if image_shape[-1] == 1:
        x = tf.image.grayscale_to_rgb(x)

    x = backbone(x)
    x = pre_bifpn(x)
    bifpn_output = bifpn(x)
    box_pred_out = box_pred(bifpn_output)
    cls_pred_out = cls_pred(bifpn_output)
    heads_out = tf.concat(values=[box_pred_out, cls_pred_out], axis=-1)

    effdet_model = tf.keras.Model(inputs=inp,
                                  outputs=heads_out,
                                  name=f"efficientdet-d{version}")

    return effdet_model


def _create_backbone(backbone_num, input_shape):
    backbone_builder = getattr(tf.keras.applications, f"EfficientNetB{backbone_num}")
    backbone = backbone_builder(input_shape=input_shape, include_top=False)
    logger.info(f"using backbone: {backbone.name}")
    return backbone


def _extract_layers(eff_net):

    layers = list(filter(lambda x: _LAYERS_PATTERN.match(x.name), eff_net.layers))
    relevant_layers = []

    for i in range(len(layers) - 1):
        if layers[i].name[:6] != layers[i + 1].name[:6]:
            relevant_layers.append(layers[i].output)

    # get last activation layer
    relevant_layers.append(eff_net.get_layer("top_activation").output)

    logger.info("extracted following layers: {}".format(", ".join(
        map(lambda l: l.name, relevant_layers))))

    return relevant_layers


def _get_image_size(
    image_size: Union[Sequence[Number], int],
    efficientdet_version: int,
) -> Sequence[int]:

    if image_size is None:
        return (
            IMAGE_RESOLUTION[efficientdet_version],
            IMAGE_RESOLUTION[efficientdet_version],
            3,
        )
    elif isinstance(image_size, Sequence):
        return image_size
    elif isinstance(image_size, int):
        return (image_size, image_size)

    raise Exception("Invalid image size parameter")
