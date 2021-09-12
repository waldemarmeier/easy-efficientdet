from abc import abstractmethod
from typing import Any, Dict, Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPool2D,
    SeparableConv2D,
    UpSampling2D,
)


class WeightedFusion(keras.layers.Layer):
    """
    @todo: add softmax fusion
    Fast weighted fusion layer
    inspired by 
    https://github.com/tensorflow/models/blob/master/research/object_detection/utils/bifpn_utils.py # noqa E501
    """
    def __init__(self, name=None, epsilon=1e-4):
        super().__init__(name=name)
        self.epsilon = epsilon

    def build(self, input_shape: Sequence[tf.Tensor]) -> None:
        self.num_inputs = len(input_shape)
        self.w = self.add_weight(
            name="fast_fusion_weight",
            shape=(self.num_inputs, 1),
            trainable=True,
            initializer=keras.initializers.constant(1 / self.num_inputs),
        )

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        w = keras.activations.relu(self.w)
        normalizer = tf.reduce_sum(w) + self.epsilon
        normalized_w = w / normalizer
        fusioned_inputs = tf.squeeze(
            tf.linalg.matmul(tf.stack(inputs, axis=-1), normalized_w),
            axis=[-1],
        )
        return fusioned_inputs

    def get_config(self):
        return {"name": self.name, "epsilon": self.epsilon}


class ChangeDim(keras.layers.Layer):
    """
    Test if activation after BN might be useful\
    @todo find out how to initialize and normalize the conv layer here
    @todo find out if activation might be useful here
    """
    def __init__(
        self,
        num_w_bifpn: int,
        name: str = "change_dim",
        bn_momentum: float = 0.99,
        bn_epsilon: float = 1e-3,
    ):
        super().__init__(name=name)
        self.num_w_bifpn = num_w_bifpn

        # BN params
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        # layers
        self.chdim_1x1_conv = Conv2D(
            filters=num_w_bifpn,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            name="chdim_1x1_conv",
        )
        self.chdim_bn = BatchNormalization(
            momentum=bn_momentum,
            epsilon=bn_epsilon,
        )
        # self.actv = Swish(name="chdim_actv")

    def call(self, inputs, training=False):
        x = self.chdim_1x1_conv(inputs)
        x = self.chdim_bn(x, training=training)
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_w_bifpn": self.num_w_bifpn,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon
        }


class PreBiFPN(keras.layers.Layer):
    """
    This layer is applied to the output of the efficient net model.
    As input, it gets levels 3,4 and 5 from efficient net
    It does following thing
        - generates two new level (6 and 7)
        - changes number of channels to the to the provided number using 1x1
            convolutions
    """
    def __init__(
        self,
        num_w_bifpn,
        name="pre_bifpn",
        bn_momentum=0.99,
        bn_epsilon=1e-3,
    ):
        super().__init__(name=name + f"_to_{num_w_bifpn}_chnlls")
        self.num_w_bifpn = num_w_bifpn
        # BN params
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        self.maxpool_P6out = MaxPool2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="same",
            name="dws_P6out",
        )
        self.dws_P7out = MaxPool2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding="same",
            name="dws_P7out",
        )

        # layers for adjusting the number of channels
        self.chdims_P3out = self._default_changedim_layer(name="chdims_P3out")
        self.chdims_P4out = self._default_changedim_layer(name="chdims_P4out")
        self.chdims_P5out = self._default_changedim_layer(name="chdims_P5out")
        self.chdims_P6out = self._default_changedim_layer(name="chdims_P6out")

    def call(self, inputs, training=False):
        P3_in, P4_in, P5_in = inputs

        # P3_out to P5_out (pre BiFPN)
        P3_out = self.chdims_P3out(P3_in, training=training)
        P4_out = self.chdims_P4out(P4_in, training=training)
        P5_out = self.chdims_P5out(P5_in, training=training)

        # P6_out (pre BiFPN)
        P6_out = self.chdims_P6out(P5_in, training=training)
        P6_out = self.maxpool_P6out(P6_out)

        # P7_out (pre BiFPN)
        P7_out = self.dws_P7out(P6_out)

        return (
            P3_out,
            P4_out,
            P5_out,
            P6_out,
            P7_out,
        )

    def _default_changedim_layer(self, name: str) -> ChangeDim:
        return ChangeDim(
            num_w_bifpn=self.num_w_bifpn,
            name=name,
            bn_momentum=self.bn_momentum,
            bn_epsilon=self.bn_epsilon,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_w_bifpn": self.num_w_bifpn,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon
        }


class SeparableConvBlock(keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        bn_momentum=0.99,
        bn_epsilon=1e-3,
        name: str = None,
        conv_kwargs={},
    ):
        super().__init__(name=name)
        # conv params
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

        # BN params
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        # layers
        # add naming convention with tf.name_scope...
        self.sep_conv2d = SeparableConv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            name="sep_conv",
            use_bias=use_bias,
            **conv_kwargs,
        )
        self.bn = BatchNormalization(
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            name="sep_conv_bn",
        )
        # @todo: try if it works better without activation after
        # - Done:
        # - Results: does not improve the performance (even reduces it a bit), BUT
        #            the val-loss is significantly more stable during training
        # - Idea: try leaky-relu here
        # self.sw_actv = Swish(name="swish_actv_post_sep_conv")

    def call(self, inputs, training=False) -> tf.Tensor:
        x = self.sep_conv2d(inputs)
        x = self.bn(x, training=training)
        # x = self.sw_actv(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        return {
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "use_bias": self.use_bias,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon
        }


class BiFPN(keras.layers.Layer):
    """
    to further improve the efficiency, we use depthwise separable convolution [7, 36]
    for feature fusion, and add batch normalization and activation
    after each convolution.
    """
    def __init__(
        self,
        num_bifpn_layers,
        num_w_bifpn,
        bn_momentum=0.99,
        bn_epsilon=1e-3,
        name=None,
    ):
        super().__init__(name=name)
        self.num_bifpn_layers = num_bifpn_layers
        self.num_w_bifpn = num_w_bifpn

        # BN params
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        # @todo add property getter  _bifpn_layers
        # idk if this a good practice
        for i in range(self.num_bifpn_layers):
            bifpn_layer = BiFPNLayer(
                bifpn_layer_num=i,
                num_w_bifpn=self.num_w_bifpn,
                bn_momentum=self.bn_momentum,
                bn_epsilon=self.bn_epsilon,
            )
            setattr(self, f"_bifpn_layer{i}", bifpn_layer)

    def call(self, inputs, training=False):

        for i in range(self.num_bifpn_layers):
            inputs = self._bifpn_layer(i)(inputs, training=training)

        return inputs

    def _bifpn_layer(self, num):
        return getattr(self, f"_bifpn_layer{num}")

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_bifpn_layers": self.num_bifpn_layers,
            "num_w_bifpn": self.num_w_bifpn,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon,
            "name": self.name,
        }


class BiFPNLayer(keras.layers.Layer):
    r"""
    Layer architecture:
        ->  P7_in ----------> P7_out ->
                  \             Λ
                   \--------    |
                  / \        \  |
        ->  P6_in -> P6_td -> P6_out ->
                  \    |        Λ
                   \---|----    |
                  / \  V     \  |
        ->  P5_in -> P5_td -> P5_out ->
                  \    |        Λ
                   \---|---     |
                  / \  V    \   |
        ->  P4_in -> P4_td -> P4_out ->
                          \     Λ
                           \    |
                            \   |
        ->  P3_in ----------> P3_out ->
    """
    def __init__(self,
                 bifpn_layer_num: int,
                 num_w_bifpn: int,
                 name: str = None,
                 bn_momentum: float = 0.99,
                 bn_epsilon: float = 1e-3,
                 mp_pool_size: int = 3,
                 mp_strides: int = 2,
                 mp_padding: str = "same"):
        super().__init__(name=name)
        self.bifpn_layer_num = bifpn_layer_num
        self.num_w_bifpn = num_w_bifpn
        # BN params
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        # max pool params
        self.mp_pool_size = mp_pool_size
        self.mp_strides = mp_strides
        self.mp_padding = mp_padding

        # P6_td
        self.ups_P7in = UpSampling2D(name="ups_P7in")
        self.wf_P7U_P6in = WeightedFusion(name="wf_P7U_P6_in")
        self.sep_conv_P6td = self._default_sep_conv(name="sep_conv_P6td")

        # P5_td
        self.ups_P6td = UpSampling2D(name="ups_P6td")
        self.wf_P6U_P5in = WeightedFusion(name="wf_P6U_P5in")
        self.sep_conv_P5td = self._default_sep_conv(name="sep_conv_P5td")

        # P4_td
        self.ups_P5td = UpSampling2D(name="ups_P5_td")
        self.wf_P5U_P4in = WeightedFusion(name="wf_P5U_P4in")
        self.sep_conv_P4td = self._default_sep_conv(name="sep_conv_P4td")

        # P3_out
        self.ups_P4td = UpSampling2D(name="ups_P4td")
        self.wf_P4U_P3in = WeightedFusion(name="wf_P4U_P3in")
        self.sep_conv_P3out = self._default_sep_conv(name="sep_conv_P3out")

        # P4_out
        self.dws_P3out = self._default_maxpool(name="dws_P3out")
        self.wf_P3D_P4td_P5in = WeightedFusion(name="wf_P3D_P4td_P5in")
        self.sep_conv_P4out = self._default_sep_conv(name="sep_conv_P4out")

        # P5_out
        self.dws_P4out = self._default_maxpool(name="dws_P4out")
        self.wf_P4D_P5td_P5in = WeightedFusion(name="wf_P4D_P5td_P5in")
        self.sep_conv_P5out = self._default_sep_conv(name="sep_conv_P5out")

        # P6_out
        self.dws_P5out = self._default_maxpool(name="dws_P5out")
        self.wf_P5D_P6td_P6in = WeightedFusion(name="wf_P5D_P6td_P6in")
        self.sep_conv_P6out = self._default_sep_conv(name="sep_conv_P6out")

        # P7_out
        self.dws_P6out = self._default_maxpool(name="dws_P6out")
        self.wf_P6D_P7in = WeightedFusion(name="wf_P6D_P7in")
        self.sep_conv_P7out = self._default_sep_conv(name="sep_conv_P7out")

    def _default_depthw_conv_init(
            self,
            mean: float = 0.0,
            stddev: float = 0.03,
            l2: float = 0.00004,  # TODO include params into constructor
    ):
        return {
            "depthwise_initializer":
            keras.initializers.TruncatedNormal(mean=mean, stddev=stddev),
            "pointwise_initializer":
            keras.initializers.TruncatedNormal(mean=mean, stddev=stddev),
            "depthwise_regularizer":
            keras.regularizers.L2(l2=l2),
            "pointwise_regularizer":
            keras.regularizers.L2(l2=l2),
        }

    def _default_sep_conv(self, name: str) -> SeparableConv2D:
        return SeparableConvBlock(
            filters=self.num_w_bifpn,
            name=name,
            bn_momentum=self.bn_momentum,
            bn_epsilon=self.bn_epsilon,
            conv_kwargs=self._default_depthw_conv_init(),
        )

    def _default_maxpool(self, name: str) -> MaxPool2D:
        return MaxPool2D(
            pool_size=self.mp_pool_size,
            strides=self.mp_strides,
            padding=self.mp_padding,
            name=name,
        )

    def call(self,
             inputs: Sequence[tf.Tensor],
             training: bool = False) -> Sequence[tf.Tensor]:
        P3_in, P4_in, P5_in, P6_in, P7_in = inputs

        # down down down
        # P6_td
        P7_U = self.ups_P7in(P7_in)
        P6_td = self.wf_P7U_P6in([P7_U, P6_in])
        P6_td = tf.nn.swish(P6_td)
        P6_td = self.sep_conv_P6td(P6_td, training=training)

        # P5_td
        P6_U = self.ups_P6td(P6_td)
        P5_td = self.wf_P6U_P5in([P6_U, P5_in])
        P5_td = tf.nn.swish(P5_td)
        P5_td = self.sep_conv_P5td(P5_td, training=training)

        # P4_td
        P5_U = self.ups_P5td(P5_td)
        P4_td = self.wf_P5U_P4in([P5_U, P4_in])
        P4_td = tf.nn.swish(P4_td)
        P4_td = self.sep_conv_P4td(P4_td, training=training)

        # P3_out
        P4_U = self.ups_P4td(P4_td)
        P3_out = self.wf_P4U_P3in([P4_U, P3_in])
        P3_out = tf.nn.swish(P3_out)
        P3_out = self.sep_conv_P3out(P3_out, training=training)

        # P4_out
        P3_D = self.dws_P3out(P3_out)
        P4_out = self.wf_P3D_P4td_P5in([P3_D, P4_td, P4_in])
        P4_out = tf.nn.swish(P4_out)
        P4_out = self.sep_conv_P4out(P4_out, training=training)

        # P5_out
        P4_D = self.dws_P4out(P4_out)
        P5_out = self.wf_P4D_P5td_P5in([P4_D, P5_td, P5_in])
        P5_out = tf.nn.swish(P5_out)
        P5_out = self.sep_conv_P5out(P5_out, training=training)

        # P6_out
        P5_D = self.dws_P5out(P5_out)
        P6_out = self.wf_P5D_P6td_P6in([P5_D, P6_td, P6_in])
        P6_out = tf.nn.swish(P6_out)
        P6_out = self.sep_conv_P6out(P6_out, training=training)

        # P7_out
        P6_D = self.dws_P6out(P6_out)
        P7_out = self.wf_P6D_P7in([P6_D, P7_in])
        P7_out = tf.nn.swish(P7_out)
        P7_out = self.sep_conv_P7out(P7_out, training=training)

        return (
            P3_out,
            P4_out,
            P5_out,
            P6_out,
            P7_out,
        )

    def get_config(self) -> Dict[str, Any]:
        return {
            "bifpn_layer_num": self.bifpn_layer_num,
            "num_w_bifpn": self.num_w_bifpn,
            "name": self.name,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon,
            "mp_pool_size": self.mp_pool_size,
            "mp_strides": self.mp_strides,
            "mp_padding": self.mp_padding,
        }


class PredLayer(keras.layers.Layer):

    CONV_PREFIX = "conv_layer"

    def __init__(
        self,
        name: str,
        width: int,
        depth: int,
        num_anchors: int = 9,
        kernel_size: Sequence[int] = (3, 3),
        strides: Sequence[int] = (1, 1),
        padding: str = "same",
        bn_momentum: float = 0.99,
        bn_epsilon: float = 1e-3,
    ):
        super().__init__(name=name)
        # hyper params
        self.width = width
        self.depth = depth
        # BN params
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        # conv params
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = "same"
        # anchors
        self.num_anchors = num_anchors

        # generate conv layers
        for i in range(depth):
            sep_conv_layer = SeparableConv2D(
                filters=width,
                padding=padding,
                strides=strides,
                kernel_size=kernel_size,
                name=f"{self.name}_conv_block{i}",
                bias_initializer="zeros",
                depthwise_initializer=keras.initializers.VarianceScaling(),
                pointwise_initializer=keras.initializers.VarianceScaling(),
                depthwise_regularizer=keras.regularizers.L2(l2=0.00004),
                pointwise_regularizer=keras.regularizers.L2(l2=0.00004),
            )
            setattr(self, f"{self.CONV_PREFIX}{i}", sep_conv_layer)

        # generate BN layers
        # seprate BN layers for every level of the feature pyramid
        # is this really necessary?
        for i in range(depth):
            for j in range(3, 8):  # from 3,..,7 for every level in FPN ouput
                bn = BatchNormalization(
                    momentum=bn_momentum,
                    epsilon=bn_epsilon,
                    name=f"bn_{self.name}_num{i}_level{j}",
                )
                setattr(self, f"bn{i}_level{j}", bn)

        self.concat = keras.layers.Concatenate(axis=1, name=f"concat_{self.name}")

    @abstractmethod
    def call(self, inputs, training=False):
        pass

    def get_conv_layer(self, num):
        return getattr(self, f"{self.CONV_PREFIX}{num}")

    def get_bn(self, num, level):
        return getattr(self, f"bn{num}_level{level}")


class BoxPredLayer(PredLayer):
    """
    Add option for conv vs. separable conv
    """
    def __init__(
        self,
        width,
        depth,
        num_anchors=9,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        bn_momentum=0.99,
        bn_epsilon=1e-3,
        name="box_predict",
    ):
        super().__init__(
            name=name,
            width=width,
            depth=depth,
            num_anchors=num_anchors,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
        )

        # 4 values for x, y, w, h
        num_values = 4
        self.num_values = num_values

        self.head = SeparableConv2D(
            filters=num_anchors * num_values,
            name="box_pred_final_conv",
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            depthwise_initializer=keras.initializers.VarianceScaling(),
            pointwise_initializer=keras.initializers.VarianceScaling(),
            bias_initializer="zeros",
        )

        self.reshape = keras.layers.Reshape((-1, num_values))

    def call(self, inputs, training=False):

        outputs = []

        # iterate through fpn levels
        for idx, fpn_level in enumerate(inputs):
            # pass through every conv layer
            for d in range(self.depth):
                fpn_level = self.get_conv_layer(d)(fpn_level)
                fpn_level = self.get_bn(d, idx + 3)(fpn_level, training=training)
                fpn_level = tf.nn.swish(fpn_level)

            fpn_level = self.head(fpn_level)
            fpn_level = self.reshape(fpn_level)
            outputs.append(fpn_level)

        outputs = self.concat(outputs)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "depth": self.depth,
            "num_anchors": self.num_anchors,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon,
            "name": self.name,
        }


class ClassPredLayer(PredLayer):
    def __init__(
        self,
        width: int,
        depth: int,
        num_anchors: int = 9,
        num_cls: int = 4,  # TODO rename to num_cls
        kernel_size: Sequence[int] = (3, 3),
        strides: Sequence[int] = (1, 1),
        padding: str = "same",
        bn_momentum: float = 0.99,
        bn_epsilon: float = 1e-3,
        name: str = "box_pred",
    ):
        super().__init__(
            width=width,
            depth=depth,
            num_anchors=num_anchors,
            name=name,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
        )
        # params
        self.width = width
        self.num_cls = num_cls
        # BN params
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        self.head = SeparableConv2D(
            filters=num_anchors * num_cls,
            name="cls_pred_final_conv",
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            bias_initializer=keras.initializers.Constant(-4.6),
            depthwise_initializer=keras.initializers.VarianceScaling(),
            pointwise_initializer=keras.initializers.VarianceScaling(),
        )
        self.reshape = keras.layers.Reshape((-1, num_cls), name="cls_pred_reshape")

    def call(self,
             inputs: Sequence[tf.Tensor],
             training: bool = False) -> Sequence[tf.Tensor]:

        outputs = []

        # iterate through fpn levels
        for idx, fpn_level in enumerate(inputs):
            # pass through every conv layer
            for d in range(self.depth):
                fpn_level = self.get_conv_layer(d)(fpn_level)
                fpn_level = self.get_bn(d, idx + 3)(fpn_level, training=training)
                fpn_level = tf.nn.swish(fpn_level)

            fpn_level = self.head(fpn_level)
            fpn_level = self.reshape(fpn_level)
            # @todo: depending on loss function might be used again
            # fpn_level = self._sigm_actv(fpn_level) # this is probably not really
            # necessary

            outputs.append(fpn_level)

        outputs = self.concat(outputs)

        return outputs

    def get_config(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "depth": self.depth,
            "num_anchors": self.num_anchors,
            "num_cls": self.num_cls,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "padding": self.padding,
            "bn_momentum": self.bn_momentum,
            "bn_epsilon": self.bn_epsilon,
            "name": self.name,
        }
