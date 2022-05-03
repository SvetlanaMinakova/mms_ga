
from DSE.low_memory.dp_by_parts import reset_phases
from converters.dnn_to_csdf import dnn_to_csfd_one_to_one
from DSE.low_memory.dp_by_parts import annotate_layers_with_phases
from simulation.csdf_simulation import simulate_execution_asap
from models.csdf_model.csdf import check_csdfg_consistency
from DSE.low_memory.buf_reuse_from_simulation import build_csdfg_reuse_buffers_from_sim_trace, minimize_csdfg_buf_sizes, \
    reuse_buffers_among_csdf
from models.data_buffers import build_naive_csdfg_buffers

"""
Build MMS (max-memory-save) buffers that employ reuse of data within (data-processing-by-parts)
and among (buffers reuse) different layers
"""
###########
# Interface


##########################################
# single dnn with no pipeline parallelism


def get_mms_buffers(dnn, phases_per_layer):
    """ Get data processing by parts + reuse buffers"""
    annotate_layers_with_phases(dnn, phases_per_layer)
    annotate_with_sim_time(dnn)

    csdf = dnn_to_csfd_one_to_one(dnn)
    consistency = check_csdfg_consistency(csdf, verbose=True)

    # build naive (non-reuse) buffers
    csdf_buffers = build_naive_csdfg_buffers(csdf)
    sim_trace = simulate_execution_asap(csdf, csdf_buffers,
                                        max_samples=1,
                                        proc_num=1,
                                        trace_memory_access=True,
                                        verbose=False)
    sim_trace.sort_tasks_by_start_time()
    minimize_csdfg_buf_sizes(sim_trace, csdf_buffers)
    reuse_dp_csdf_buffers = build_csdfg_reuse_buffers_from_sim_trace(sim_trace, csdf_buffers)

    reset_phases(dnn)
    return reuse_dp_csdf_buffers

######################################
# single dnn with pipeline parallelism


def get_mms_buffers_pipelined(dnn_partitions, phases_per_layer_per_dnn):
    """ Get data processing by parts + reuse buffers"""
    # create buffers within partitions
    csdf_buffers_per_partition = []
    dp_reuse_buffers = []

    for partition in dnn_partitions:
        annotate_layers_with_phases(partition, phases_per_layer_per_dnn[partition.name])
        annotate_with_sim_time(partition)

        csdf = dnn_to_csfd_one_to_one(partition)
        consistency = check_csdfg_consistency(csdf, verbose=True)

        # build naive (non-reuse) buffers
        csdf_buffers = build_naive_csdfg_buffers(csdf)
        sim_trace = simulate_execution_asap(csdf, csdf_buffers,
                                            max_samples=1,
                                            proc_num=1,
                                            trace_memory_access=True,
                                            verbose=False)
        sim_trace.sort_tasks_by_start_time()
        minimize_csdfg_buf_sizes(sim_trace, csdf_buffers)
        reuse_csdf_buffers = build_csdfg_reuse_buffers_from_sim_trace(sim_trace, csdf_buffers)
        csdf_buffers_per_partition.append(reuse_csdf_buffers)

    # NOTE: buffers CANNOT be reused among different partitions executed on different processors!
    # in a CNN executed as a pipeline, buffers can only be reused within every partition
    # Thus, the result buffers are a superset of buffers per-partition
    for buf_list_per_partition in csdf_buffers_per_partition:
        for buf in buf_list_per_partition:
            # rename buffers to give them unique names
            buf.name = "B" + str(len(dp_reuse_buffers))
            dp_reuse_buffers.append(buf)

    # reset number of phases per partition
    for partition in dnn_partitions:
        reset_phases(partition)

    return dp_reuse_buffers


########################################
# multi-dnn with no pipeline parallelism

def set_auto_buffer_names(csdf_buffers):
    """
    Set auto-names to CSDF buffers
    :param csdf_buffers: CSDF buffers
    """
    buf_id = 0
    for buf in csdf_buffers:
        # rename buffer
        buf.name = "B" + str(buf_id)
        buf_id += 1


def get_mms_buffers_multi(dnns, phases_per_layer_per_dnn):
    """ Get buffers with data processing by parts where memory reused within and among dnns"""
    buffers_per_dnn = []

    for dnn in dnns:
        csdf_buffers = get_mms_buffers(dnn, phases_per_layer_per_dnn[dnn.name])
        buffers_per_dnn.append(csdf_buffers)

    # reuse buffers among dnn (csdf)
    shared_buffers = reuse_buffers_among_csdf(buffers_per_dnn)
    set_auto_buffer_names(shared_buffers)
    return shared_buffers


########################################
# multi-dnn with no pipeline parallelism

def get_mms_buffers_multi_pipelined(partitions_per_dnn: [], phases_per_layer_per_partition_per_dnn: []):
    """
    Get buffers with data processing by parts where memory reused within and among dnns
    :param partitions_per_dnn: list [partitions_1, partitions_2, ..., partitionsN] where
    partitions_i is a list of partitions of a DNN, N is the total number of DNNs
    :param phases_per_layer_per_partition_per_dnn: list
    [phases_per_partition_1, phases_per_partition_2, ..., phases_per_partition_N] where
    phases_per_partition_j is a dictionary with key = dnn (partition) name, value = dictionary
    with phases (values) per dnn layer (keys)
    :return: buffers shared among dnns, where some of dnns are executed with pipeline parallelism
    """
    buffers_per_dnn = []

    for dnn_id in range(len(partitions_per_dnn)):
        partitions = partitions_per_dnn[dnn_id]
        phases_per_layer_per_partition = phases_per_layer_per_partition_per_dnn[dnn_id]

        # single-partition dnn (executed sequentially)
        if len(partitions) == 1:
            single_partition = partitions[0]
            phases_per_layer = phases_per_layer_per_partition[single_partition.name]
            dnn_buffers = get_mms_buffers(single_partition, phases_per_layer)
        # multi-partition dnn (executed as a pipeline)
        else:
            dnn_buffers = get_mms_buffers_pipelined(partitions, phases_per_layer_per_partition)

        buffers_per_dnn.append(dnn_buffers)

    # reuse buffers among dnns (csdf)
    shared_buffers = reuse_buffers_among_csdf(buffers_per_dnn)
    set_auto_buffer_names(shared_buffers)

    return shared_buffers


def annotate_with_sim_time(dnn):
    """
    Annotate layers with fake time to simulate their schedule
    # NOTE: this is NECESSARY for timed simulation!
    """
    for layer in dnn.get_layers():
        layer.time_eval = max(layer.phases, 1)

