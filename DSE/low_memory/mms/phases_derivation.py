from DSE.low_memory.dp_by_parts import get_max_phases_per_layer
from models.dnn_model.dnn import DNN

"""
Determines number of phases, performed by every layer in a DNN/multiple DNNs, using
data processing by parts, encoded in a binary string of length M, where M = total number 
of layers in all the DNNs, used by the application. Every i-th, 0<i<N element in the chromosome 
encodes data processing by parts, exploited by i-th DNN layer in a DNN-based application. '
Layers are indexed in execution order (from input layer to output layer). If an application uses 
multiple DNNs, layers of the DNNs are concatenated in the order, in which DNNs are mentioned
in the application (in the input dnns list)
"""


def get_phases_per_layer(dnn: DNN, dp_encoding: [bool], max_phases_per_layer=None) -> {}:
    """
    Determine maximum number of phases performed by every layer of a DNN
    :param dp_encoding: ata processing by parts, encoded in a binary string
        (see explanation at the beginning of the script
    :param max_phases_per_layer: maximum number of phases per DNN layer
    :param dnn: DNN
    """
    if max_phases_per_layer is None:
        max_phases = get_max_phases_per_layer(dnn)
    else:
        max_phases = max_phases_per_layer
    
    phases_per_layer = {}
    layer_id_in_encoding = 0
    for layer in dnn.get_layers():
        max_layer_phases = max_phases[layer.name]
        layer_phases = decode_layer_phases(max_layer_phases, layer_id_in_encoding, dp_encoding)
        max_phases[layer.name] = layer_phases
        layer_id_in_encoding += 1

    return phases_per_layer


def decode_layer_phases(max_layer_phases: int, layer_id_in_encoding: int, dp_encoding: [bool]) -> int:
    """
    :param max_layer_phases: (int) maximum number of phases, performed by DNN layer
    :param dp_encoding: ata processing by parts, encoded in a binary string
        (see explanation at the beginning of the script)
    :param layer_id_in_encoding: layer id in dp_encoding
    :return: number of phases, performed by the dnn layer according to the encoding 
         and maximum phases
    """
    dp = dp_encoding[layer_id_in_encoding]
    layer_phases = max_layer_phases if dp else 1
    return layer_phases
    

def get_max_phases_per_layer_per_partition(dnn_partitions: []) -> {}:
    """  Derive maximum  number of phases per layer per partition for a single DNN
    :param dnn_partitions: list of dnn partitions (sub-networks)
    :return: dictionary, where key (string) = DNN partition name,
        value = partition_i_phases, i in [1, N], where N is total number of dnn partitions (sub-networks),
        dnn_phases_i, i in [1, N] is a dictionary with maximum phases of i-th DNN partition,
        where key (str) = name of layer in i-th DNN partition,
        value (int) = maximum number of phases, performed by the layer in i-th DNN partition
    """
    max_phases_per_layer_per_partition = {}
    for partition in dnn_partitions:
        max_phases_per_layer = get_max_phases_per_layer(partition)
        max_phases_per_layer_per_partition[partition.name] = max_phases_per_layer
    return max_phases_per_layer_per_partition


def get_phases_per_layer_per_partition(dnn_partitions: [],
                                       dp_encoding: [bool],
                                       max_phases_per_layer_per_partition=None) -> {}:
    """
    Derive number of phases per layer per partition for a single DNN
    :param dnn_partitions: list of dnn partitions (sub-networks)
    :param dp_encoding: ata processing by parts, encoded in a binary string
        (see explanation at the beginning of the script\
    :param max_phases_per_layer_per_partition: maximum number of phases per layer per dnn partition.
    If unspecified (is None), is computed automatically
    :return: phases_per_layer_per_partition: dictionary, where key (string) = DNN partition name,
        value = partition_i_phases, i in [1, N], where N is total number of dnn partitions (sub-networks),
        dnn_phases_i, i in [1, N] is a dictionary with phases of i-th DNN partition,
        where key (str) = name of layer in i-th DNN partition,
        value (int) = number of phases, performed by the layer in i-th DNN partition
    """
    if max_phases_per_layer_per_partition is None:
        max_phases = get_max_phases_per_layer_per_partition(dnn_partitions)
    else:
        max_phases = max_phases_per_layer_per_partition

    phases_per_layer_partition = {}
    layer_id_in_encoding = 0
    for partition in dnn_partitions:
        max_phases_per_layer = max_phases[partition.name]
        phases_per_layer = {}
        for layer in partition.get_layers():
            max_layer_phases = max_phases_per_layer[layer.name]
            layer_phases = decode_layer_phases(max_layer_phases, layer_id_in_encoding, dp_encoding)
            phases_per_layer[layer.name] = layer_phases
            layer_id_in_encoding += 1
        phases_per_layer_partition[partition.name] = phases_per_layer


def get_max_phases_per_layer_per_dnn(dnns: [DNN]) -> {}:
    """
    Determine maximum number of phases performed by every layer of every DNN in input DNNs list
    :param dnns: input DNNs list
    :return max phases per layer per dnn
    """
    max_phases = {}
    for dnn in dnns:
        max_dnn_phases = get_max_phases_per_layer(dnn)
        max_phases[dnn.name] = max_dnn_phases
    return max_phases


def get_phases_per_layer_per_dnn(dnns: [DNN], dp_encoding: [bool], max_phases_per_layer_per_dnn=None) -> {}:
    """
    Determine number of phases performed by every layer of every DNN in input DNNs list
    :param dnns: input DNNs list
    :param dp_encoding: ata processing by parts, encoded in a binary string
        (see explanation at the beginning of the script
    :param max_phases_per_layer_per_dnn: maximum number of phases per layer per DNN
        if unspecified (is None), is computed automatically
    :return max phases per layer per dnn
    """
    if max_phases_per_layer_per_dnn is None:
        max_phases = get_max_phases_per_layer_per_dnn(dnns)
    else:
        max_phases = max_phases_per_layer_per_dnn
    
    phases = {}
    layer_id_in_encoding = 0
    for dnn in dnns:
        max_phases_per_layer = max_phases[dnn.name]
        phases_per_layer = {}
        for layer in dnn.get_layers():
            max_layer_phases = max_phases_per_layer[layer.name]
            layer_phases = decode_layer_phases(max_layer_phases, layer_id_in_encoding, dp_encoding)
            phases_per_layer[layer.name] = layer_phases
            layer_id_in_encoding += 1
        phases[dnn.name] = phases_per_layer
    return phases
        

def get_max_phases_per_layer_per_partition_per_dnn(partitions_per_dnn: []):
    """
    Determine maximum number of phases performed by every DNN layer
    :param partitions_per_dnn: list of pipelined partitions (sub-networks) per dnn
    """
    max_phases_per_layer_per_partition_per_dnn = []
    for dnn_partitions in partitions_per_dnn:
        max_phases_per_layer_per_partition = get_max_phases_per_layer_per_partition(dnn_partitions)
        max_phases_per_layer_per_partition_per_dnn.append(max_phases_per_layer_per_partition)
    return max_phases_per_layer_per_partition_per_dnn


def get_phases_per_layer_per_partition_per_dnn(partitions_per_dnn: [],
                                               dp_encoding: [bool],
                                               max_phases_per_layer_per_partition_per_dnn=None) -> []:
    """
    Determine maximum number of phases performed by every DNN layer
    :param partitions_per_dnn: list of pipelined partitions per dnn
    :param dp_encoding: ata processing by parts, encoded in a binary string
        (see explanation at the beginning of the script
    :param max_phases_per_layer_per_partition_per_dnn: maximum number of phases per partition per DNN
        if unspecified (is None), is computed automatically
    """
    if max_phases_per_layer_per_partition_per_dnn is None:
        max_phases = get_max_phases_per_layer_per_partition_per_dnn(partitions_per_dnn)
    else:
        max_phases = max_phases_per_layer_per_partition_per_dnn

    phases_per_layer_per_partition_per_dnn = []
    dnns_num = len(partitions_per_dnn)
    layer_id_in_chromosome = 0
    for dnn_id in range(dnns_num):
        max_ph_per_dnn = max_phases[dnn_id]
        ph_per_dnn = {}
        partitions = partitions_per_dnn[dnn_id]
        for partition in partitions:
            max_ph_per_dnn_per_partition = max_ph_per_dnn[partition.name]
            ph_per_dnn_per_partition = {}
            for layer in partition.get_layers():
                max_layer_phases = max_ph_per_dnn_per_partition[layer.name]
                layer_phases = decode_layer_phases(max_layer_phases, layer_id_in_chromosome, dp_encoding)
                ph_per_dnn_per_partition[layer.name] = layer_phases
                layer_id_in_chromosome += 1
            ph_per_dnn[partition.name] = ph_per_dnn_per_partition
        phases_per_layer_per_partition_per_dnn.append(ph_per_dnn)

    return phases_per_layer_per_partition_per_dnn

