from models.dnn_model.dnn import DNN
from eval.memory.csdf_model_mem_eval import eval_csdf_buffers_memory_mb
from DSE.low_memory.mms.buf_building import get_mms_buffers_no_pipeline, get_mms_buffers_multi_pipelined

############################
# Eval functions for MMS-GA


def eval_chromosome_time_loss_ms(phases_per_layer, delay_per_phase_ms=0.0005):
    """
    Compute time loss (delay), caused by data processing by parts: the smaller, the better
    :param delay_per_phase_ms: sync. delay per one extra phase
    :param phases_per_layer: dictionary where key = name of layer in the DNN, value=
    number of phases performed by every layer of a DNN
    :return: delay in ms
    """
    extra_phases = 0

    for phases_num in phases_per_layer.values():
        extra_phases_in_layer = max(phases_num-1, 0)
        extra_phases += extra_phases_in_layer

    delay = extra_phases * delay_per_phase_ms
    return delay


def eval_chromosome_time_loss_ms_multi_pipeline(phases_per_layer_per_partition_per_dnn: [], delay_per_phase_ms=0.0005):
    """
    Compute time loss (delay), caused by data processing by parts: the smaller, the better
    :param delay_per_phase_ms: sync. delay per one extra phase
    :param phases_per_layer_per_partition_per_dnn:
    :return: delay in ms
    """
    extra_phases = 0
    for phases_per_partition in phases_per_layer_per_partition_per_dnn:
        for phases_per_layer in phases_per_partition.values():
            for phases_num in phases_per_layer.values():
                extra_phases_in_layer = max(phases_num-1, 0)
                extra_phases += extra_phases_in_layer

    delay = extra_phases * delay_per_phase_ms
    return delay


def eval_dnn_buffers_size_mb(dnn: DNN, phases_per_layer, data_token_size=4):
    """
    Eval DNN memory in megabytes with max-mem-save (DP + reuse) memory reduction: the smaller, the better
    :param dnn: DNN to eval buffers of
    :param phases_per_layer: dictionary where key = name of layer in the DNN, value=
    number of phases performed by every layer of a DNN
    :param data_token_size: size of one data token (in Bytes)
    :return: size of DNN buffers (in MB)
    """

    # build buffers
    mms_csdf_buffers = get_mms_buffers_no_pipeline(dnn, phases_per_layer)

    # eval buffers size
    buf_size = eval_csdf_buffers_memory_mb(mms_csdf_buffers, data_token_size)
    return buf_size


def eval_dnn_buffers_size_multi_pipelined_mb(partitions_per_dnn: [],
                                             phases_per_layer_per_partition_per_dnn: [],
                                             data_token_size=4):
    """
    Eval DNN memory in megabytes with max-mem-save (DP + reuse) memory reduction: the smaller, the better
    :param partitions_per_dnn: list [partitions_1, partitions_2, ..., partitionsN] where
    partitions_i is a list of partitions of a DNN, N is the total number of DNNs
    :param phases_per_layer_per_partition_per_dnn: list
    [phases_per_partition_1, phases_per_partition_2, ..., phases_per_partition_N] where
    phases_per_partition_j is a dictionary with key = dnn (partition) name, value = dictionary
    with phases (values) per dnn layer (keys)
    :param data_token_size: size of one data token (in Bytes)
    :return: size of DNN buffers (in MB)
    """

    # build buffers
    mms_csdf_buffers = get_mms_buffers_multi_pipelined(partitions_per_dnn, phases_per_layer_per_partition_per_dnn)

    # eval buffers size
    buf_size = eval_csdf_buffers_memory_mb(mms_csdf_buffers, data_token_size)
    return buf_size

