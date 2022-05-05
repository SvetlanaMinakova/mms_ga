"""
Simulate data processing by parts, used to reduce CNN memory and considered in the
dnn memory evaluation
"""

from models.dnn_model.dnn import DNN, Layer


def reset_phases(dnn: DNN):
    """
    Set execution of every layer in one phase
    :param dnn: dnn
    """
    for layer in dnn.get_layers():
        layer.phases = 1


def annotate_layers_with_phases(dnn: DNN, phases_per_layer: {}):
    """
    Annotate dnn layers with phases
    :param dnn: dnn
    :param phases_per_layer: phases per dnn layer
    """
    for layer in dnn.get_layers():
        if layer.name in phases_per_layer.keys():
            layer.phases = phases_per_layer[layer.name]


def get_max_phases_per_layer(dnn: DNN) -> {}:
    """
    Compute maximum number of phases, performed by the layers of a dnn
    :param dnn: dnn
    :return: maximum number of phases, performed by the layers of the dnn
    """
    phases_per_layer = {}
    for layer in dnn.get_layers():
        max_phases = get_max_phases(dnn, layer)
        phases_per_layer[layer.name] = max_phases

    return phases_per_layer


def get_max_phases(dnn: DNN, layer: Layer):
    """
    Compute maximum number of phases, performed by the layer
    :param dnn: DNN that owns the layer
    :param layer CNN layer
    :return: maximum number of phases, performed by the layer
    """
    # default
    max_phases = layer.oh
    op = layer.op

    if op in ["gemm"]:
        max_phases = 1
        return max_phases

    # external i/os
    # TODO:refactoring
    if op == "data":
        if layer.name is not None:
            if "external" in layer.name:
                max_phases = 1
                return max_phases

    # mul-operator with broadcast input (exists in keras efficientnetb0)
    """
     # TODO:check!!!
    if layer.subop == "mul":
        layer_input_connections = dnn.get_layer_input_connections(layer)
        for connection in layer_input_connections:
            if connection.src.oh == 1 or connection.src.ow == 1:
                max_phases = 1
                return max_phases
    """

    """
    if layer.built_in:
        max_phases = 1
        return max_phases
    
    if op in ["conv", "pool"]:
        max_phases = layer.oh
        return max_phases
    
    if op in ["normalization", "arithmetic"]:
        max_phases = layer.res
        return max_phases
    """

    return max_phases


def eval_thr_loss(phases_per_dnn, throughput_loss_per_phase_ms=0.0001):
    """ evaluate throughput loss, caused by data processing by parts
    :param phases_per_dnn: dictionary, where every item has a key = dnn name, value =
    dictionary of phases per dnn layer
    :param throughput_loss_per_phase_ms: disctionary, where key = dnn name,
    throughput loss per phase/kernel launch
    :return: throughput loss per dnn
    """
    thr_losses_per_dnn = {}
    for item in phases_per_dnn.items():
        dnn_name, phases_per_layer = item
        dnn_throughput_loss_ms = 0.0
        for phases in phases_per_layer.values():
            dnn_throughput_loss_ms += (phases - 1) * throughput_loss_per_phase_ms
        thr_losses_per_dnn[dnn_name] = dnn_throughput_loss_ms

    return thr_losses_per_dnn



