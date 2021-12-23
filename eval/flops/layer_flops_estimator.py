def eval_layer_flops(layer):
    """
    Eval number of floating-point operations (FLOPs), performed by a layer
    :param layer: layer
    :return:  number of floating-point operations (FLOPs), performed by a layer
    """
    # default flops
    flops = layer.ih * layer.iw * layer.ifm
    op = layer.op
    sub_op = layer.subop
    if op in ["data", "none", "skip"]:
        flops = 0
    if op == "conv":
        flops = layer.fs * layer.fs * layer.ifm * layer.ofm
    if op == "gemm":
        flops = layer.ih * layer.iw * layer.ifm * layer.ofm
    if op == "pool":
        flops = layer.oh * layer.ow * layer.ofm
    if op in ["arithmetic", "normalization"]:
        flops = layer.ih * layer.iw * layer.ifm

    return flops