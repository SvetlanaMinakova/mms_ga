from models.dnn_model.dnn import Layer
"""
This module generates shapes of DNN parameters (weights and biases) for DNN analysis and simulation
"""


def generate_param_shapes_dict(layer: Layer, data_layout="NCHW"):
    param_shapes = {}
    add_weights_shape(param_shapes, layer, data_layout)
    add_bias_shape(param_shapes, layer)
    if layer.op == "normalization":
        add_normalization_param_shapes(param_shapes, layer)
    return param_shapes


def add_weights_shape(param_shapes: {}, layer: Layer, data_layout="NCHW"):
    weights_shape = generate_weights_shape(layer, data_layout)
    if weights_shape:
        param_shapes["weights"] = weights_shape


def add_bias_shape(param_shapes: {}, layer: Layer):
    bias_shape = generate_bias_shape(layer)
    if bias_shape:
        param_shapes["bias"] = bias_shape


def add_normalization_param_shapes(param_shapes: {}, layer: Layer):
    if layer.subop in ["bn", "batchnormalization"]:
        param_shape = layer.ofm
        param_names = ["shift", "scale", "power"]
        for param_name in param_names:
            param_shapes[param_name] = [param_shape]


def generate_weights_shape(layer: Layer, data_layout="NCHW"):
    """
    Generate shape of weights for DNN layer
    :param layer: DNN layer
    :param data_layout: tensor dims order (NCHW or NHWC)
    :return:
    """
    op = layer.op
    subop = layer.subop

    if op == "conv":
        n = layer.ifm
        c = layer.ofm
        h = w = layer.fs

        if subop == "depthwiseconv":
            c = 1

        if data_layout == "NCHW":
            return [n, c, h, w]
        else:
            return [n, h, w, c]

    if op == "gemm":
        n = layer.ifm
        c = layer.ofm
        h = layer.ih
        w = layer.iw
        if data_layout == "NCHW":
            return [n, c, h, w]
        else:
            return [n, h, w, c]

    return []


def generate_bias_shape(layer: Layer):
    """
    Generate shape of weights for DNN layer
    :param layer: DNN layer
    :return:
    """
    op = layer.op
    if op in ["conv", "gemm"]:
        return [layer.ofm]

    return []
