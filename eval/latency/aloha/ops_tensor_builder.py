"""
This module builds ops tensor from CNN layer description
"""
from models.dnn_model.dnn import Layer


def build_ops_tensor(layer: Layer):
    """
    Represent ops within layer as a 6d ops tensor
    """
    if layer.op == "conv":
        ops_tensor = {"oh": layer.oh, "ow": layer.ow, "ifm": layer.ifm, "ofm": layer.ofm, "kh": layer.fs, "kw": layer.fs}
        return ops_tensor

    if layer.op == "pool":
        ops_tensor = {"oh": layer.oh, "ow": layer.ow, "ifm": 1, "ofm": layer.ofm, "kh": layer.fs,
                      "kw": layer.fs}
        return ops_tensor

    if layer.op == "gemm":
        # TODO: check!!!
        ops_tensor = {"oh": layer.ih * layer.iw * layer.ifm, "ow": layer.ofm, "ifm": 1, "ofm": 1, "kh": 1, "kw": 1}
        # ops_tensor = {"oh": layer.ih, "ow": layer.iw, "ifm": layer.ifm, "ofm": layer.ofm, "kh": 1, "kw": 1}
        # ops_tensor = {"ow": layer.iw * layer.ih * layer.ifm, "ofm": layer.ofm, "kw": layer.fs * layer.fs}
        return ops_tensor

    # default
    ops_tensor = {"oh": layer.oh, "ow": layer.ow, "ifm": layer.ifm, "ofm": layer.ofm, "kh": layer.fs, "kw": layer.fs}
    return ops_tensor
