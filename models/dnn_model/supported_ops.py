"""
Supported operators are inherited from ONNX model
and distributed into groups by behaviour
Every operator has a "coarse-grain" operator,
which describes general operator behaviour and
sub-op, which specifies the "flavour" of the operator
for example if layer has op="gemm", it can have subop="matmul",
where "gemm" describes general operator behaviour and "matmul" specifies details
(matmul is gemm with no bias)
Analogously, a layer can have op=pool, and subop=maxpool etc.
"""


def print_supported_dnn_ops():
    supported_dnn_ops = get_supported_dnn_ops()
    for item in supported_dnn_ops.items():
        k, v = item
        print("op:", k)
        print("  sub-ops:", v)


def get_supported_dnn_ops():
    """
    Supported dnn operators and sub-operators
    """
    ops_and_subops = {}

    op = "data"
    subops = ["input_examples", "output"]
    ops_and_subops[op] = subops

    op = "none"
    subops = []
    ops_and_subops[op] = subops

    op = "conv"
    subops = ["conv", "separableconv"]
    ops_and_subops[op] = subops

    op = "gemm"
    subops = ["gemm", "fc", "matmul"]
    ops_and_subops[op] = subops

    op = "pool"
    subops = ["maxpool", "averagepool", "globalaveragepool"]
    ops_and_subops[op] = subops

    op = "normalization"
    subops = ["batchnormalization", "bn", "lrn"]
    ops_and_subops[op] = subops

    op = "arithmetic"
    subops = ["add", "mul", "div", "sub"]
    ops_and_subops[op] = subops

    op = "skip"
    subops = ["flatten", "reshape", "dropout"]
    ops_and_subops[op] = subops

    return ops_and_subops