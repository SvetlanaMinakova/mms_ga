from DSE.low_memory.dp_by_parts import reset_phases
from converters.dnn_to_csdf import dnn_to_csfd_one_to_one
from DSE.low_memory.dp_by_parts import annotate_layers_with_phases
from simulation.csdf_simulation import simulate_execution_asap
from models.csdf_model.csdf import check_csdfg_consistency
from DSE.low_memory.buf_reuse_from_simulation import build_csdfg_reuse_buffers_from_sim_trace, minimize_csdfg_buf_sizes, \
    reuse_buffers_among_csdf
from models.data_buffers import build_naive_csdfg_buffers
from DSE.low_memory.mms.phases_derivation import get_phases_per_layer, get_phases_per_layer_per_partition, \
    get_phases_per_layer_per_dnn, get_phases_per_layer_per_partition_per_dnn
from DSE.scheduling.mms_dnn_inf_model_schedule import MMSDNNInfModelSchedule, csdf_sim_trace_schedule_to_dnn_schedule,\
    copy_dnn_schedule

"""
The module builds MMS (max-memory-save) buffers that employ reuse of data within (data-processing-by-parts)
and among (buffers reuse) different layers of DNN(s), used by a DNN-based application
"""
###########
# Interface


def get_mms_buffers_and_schedule(dnns: [],
                                 partitions_per_dnn: [],
                                 dp_encoding: [],
                                 generate_schedule=False,
                                 verbose=False):
    """
    Get MMS buffers for a DNN-based application, using a one or multiple of DNNs,
        where every DNN is possibly executed as a set of pipelined partitions
    :param dnns: DNN(s) used by the application
    :param partitions_per_dnn: list [partitions_1, partitions_2, ..., partitionsN] where
        partitions_i is a list of partitions (sub-networks) of a DNN, N is the total number of DNNs
    :param dp_encoding: data processing by parts, encoded in a binary string of length M,
        where M = total number of layers in all the DNNs, used by the application
        Every i-th, 0<i<N element in the chromosome encodes data processing by parts, exploited by i-th DNN layer
        in a DNN-based application. Layers are indexed in execution order (from input layer to output layer).
        If an application uses multiple DNNs, layers of the DNNs are concatenated in the order, in which DNNs are mentioned
        in the application (in the input dnns list)
    :param generate_schedule: (flag) if yes, actual per-partition-per dnn schedule is generated. Otherwise,
        null-schedule is returned
    :param verbose: print details of buffers generation
    :return: list of MMS (data processing by parts + buffers reuse) CSDF buffers, used by the application,
        and schedule, required for the application to be executed with proposed buffers
    """
    if len(dnns) == 0:
        raise Exception("MMS buffers derivation error: empty dnns list!")

    # classify application
    # determine whether application uses a single DNN or multiple DNNs
    is_multi_dnn = len(dnns) > 1
    # determine whether application exploits pipeline parallelism
    is_pipelined = not (are_null_or_empty_partitions(partitions_per_dnn) or are_single_dnn_partitions(partitions_per_dnn))

    if verbose:
        print("Build MMS buffers for a CNN-based application (multi-dnn:", is_multi_dnn,
              ", pipelined:", is_pipelined, ")")

    # single-dnn applications
    if not is_multi_dnn:
        # no pipeline parallelism is exploited
        if not is_pipelined:
            single_dnn = dnns[0]
            phases = get_phases_per_layer(single_dnn, dp_encoding)
            buffers, schedule = get_mms_buffers_no_pipeline(single_dnn, phases, generate_schedule)
            # format schedule
            return buffers, schedule

        # pipeline parallelism is exploited
        else:
            single_dnn_partitions = partitions_per_dnn[0]
            phases = get_phases_per_layer_per_partition(single_dnn_partitions, dp_encoding)
            buffers, schedule = get_mms_buffers_pipelined(single_dnn_partitions, phases, dnns[0].name, generate_schedule)
            return buffers, schedule

    # multi-dnn applications
    else:
        # no pipeline parallelism is exploited
        if not is_pipelined:
            phases = get_phases_per_layer_per_dnn(dnns, dp_encoding)
            buffers, schedule = get_mms_buffers_multi(dnns, phases, generate_schedule)
            return buffers, schedule
        # pipeline parallelism is exploited
        else:
            phases = get_phases_per_layer_per_partition_per_dnn(partitions_per_dnn, dp_encoding)
            buffers, schedule = get_mms_buffers_multi_pipelined(partitions_per_dnn,
                                                                phases,
                                                                [dnn.name for dnn in dnns],
                                                                generate_schedule)
            return buffers, schedule


def are_null_or_empty_partitions(partitions_per_dnn: []):
    """
    Determine whether partitions are null or empty
    :param partitions_per_dnn: list [partitions_1, partitions_2, ..., partitionsN] where
        partitions_i is a list of partitions (sub-networks) of a DNN, N is the total number of DNNs
    :return: True if partitions are null or empty and False otherwise
    """
    if partitions_per_dnn is None:
        return True
    if len(partitions_per_dnn) == 0:
        return True
    for partitions in partitions_per_dnn:
        if len(partitions) > 0:
            return False
    return True


def are_single_dnn_partitions(partitions_per_dnn: []):
    """
    Determine whether partitions are a special case of partitions, where
        every partition = whole DNN
    :param partitions_per_dnn: list [partitions_1, partitions_2, ..., partitionsN] where
        partitions_i is a list of partitions (sub-networks) of a DNN, N is the total number of DNNs
    :return: True if every partition = whole DNN and false otherwise
    """
    if are_null_or_empty_partitions(partitions_per_dnn):
        return False

    for partitions in partitions_per_dnn:
        if len(partitions) != 1:
            return False

    return True


##########################################
# single dnn with no pipeline parallelism


def get_mms_buffers_no_pipeline(dnn, phases_per_layer: {}, generate_schedule=False):
    """
    Get buffers with data processing by parts and buffers reuse for a single-DNN application
    where memory reused within and among dnns and no pipeline parallelism is exploited
    :param dnn: single DNN, used by the application
    :param phases_per_layer:  dictionary with phases of the DNN, where key (str) = name of a DNN layer,
        value (int) = number of phases, performed by the layer
    :param generate_schedule: (flag) if yes, actual per-partition-per dnn schedule is generated. Otherwise,
        null-schedule is returned
    :return: list of CSDF buffers, used by the application, and schedule (object of MMSDNNInfModelSchedule class),
        required for the application to be executed with proposed buffers. If generate_schedule flag is False.
        schedule is None.
    """
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
    associate_buffers_with_csdf_model(reuse_dp_csdf_buffers, dnn.name)

    reset_phases(dnn)

    # CSDF model schedule (execution order of actors within CSDF model)
    schedule = None
    if generate_schedule:
        schedule = MMSDNNInfModelSchedule([dnn.name])
        sim_trace_schedule = sim_trace.get_asap_schedule()
        dnn_schedule = csdf_sim_trace_schedule_to_dnn_schedule(sim_trace_schedule)
        schedule.append_dnn_partition_schedule(dnn.name, dnn.name, dnn_schedule)

    return reuse_dp_csdf_buffers, schedule


######################################
# single dnn with pipeline parallelism


def get_mms_buffers_pipelined(dnn_partitions,
                              phases_per_layer_per_partition: {},
                              dnn_name: str,
                              generate_schedule=False):
    """
    Get buffers with data processing by parts and buffers reuse for a single-DNN application
    where memory reused within and among dnns and no pipeline parallelism is exploited
    :param dnn_partitions: partitions (sub-networks) of the single DNN, used by the application
    :param phases_per_layer_per_partition: dictionary, where key (string) = DNN partition name,
        value = partition_i_phases, i in [1, N], where N is total number of dnn partitions (sub-networks),
        dnn_phases_i, i in [1, N] is a dictionary with phases of i-th DNN partition,
        where key (str) = name of layer in i-th DNN partition,
        value (int) = number of phases, performed by the layer in i-th DNN partition
    :param dnn_name: name of the DNN
    :param generate_schedule: (flag) if yes, actual per-partition-per dnn schedule is generated. Otherwise,
        null-schedule is returned
    :return: list of CSDF buffers, used by the application, and schedule (object of MMSDNNInfModelSchedule class),
        required for the application to be executed with proposed buffers. If generate_schedule flag is False.
        schedule is None.
    """
    # create buffers within partitions
    csdf_buffers_per_partition = []
    dp_reuse_buffers = []

    schedule = MMSDNNInfModelSchedule([dnn_name])

    for partition in dnn_partitions:
        annotate_layers_with_phases(partition, phases_per_layer_per_partition[partition.name])
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

        if generate_schedule:
            sim_trace_partition_schedule = sim_trace.get_asap_schedule()
            partition_schedule = csdf_sim_trace_schedule_to_dnn_schedule(sim_trace_partition_schedule)
            schedule.append_dnn_partition_schedule(dnn_name, partition.name, partition_schedule)

    # NOTE: buffers CANNOT be reused among different partitions executed on different processors!
    # in a CNN executed as a pipeline, buffers can only be reused within every partition
    # Thus, the result buffers are a superset of buffers per-partition
    for buf_list_per_partition in csdf_buffers_per_partition:
        for buf in buf_list_per_partition:
            # rename buffers to give them unique names
            buf.name = "B" + str(len(dp_reuse_buffers))
            dp_reuse_buffers.append(buf)

    associate_buffers_with_csdf_model(dp_reuse_buffers, dnn_name)

    # reset number of phases per partition
    for partition in dnn_partitions:
        reset_phases(partition)

    if not generate_schedule:
        schedule = None

    return dp_reuse_buffers, schedule


########################################
# multi-dnn with no pipeline parallelism

def get_mms_buffers_multi(dnns, phases_per_layer_per_dnn: {}, generate_schedule=False):
    """
    Get buffers with data processing by parts and buffers reuse for a multi-CNN application
    where memory reused within and among dnns and no pipeline parallelism is exploited
    :param dnns: list of dnns
    :param phases_per_layer_per_dnn: dictionary, where key = i-th DNN name, value =
        dictionary with phases of i-th DNN, where key (str) = name of layer in i-th DNN,
        value (int) = number of phases, performed by the layer
    :param generate_schedule: (flag) if yes, actual per-partition-per dnn schedule is generated. Otherwise,
        null-schedule is returned
    :return: list of CSDF buffers, used by the application, and schedule (object of MMSDNNInfModelSchedule class),
        required for the application to be executed with proposed buffers. If generate_schedule flag is False.
        schedule is None.
    """
    buffers_per_dnn = []

    schedule = MMSDNNInfModelSchedule([dnn.name for dnn in dnns])

    for dnn in dnns:
        csdf_buffers, dnn_schedule = get_mms_buffers_no_pipeline(dnn,
                                                                 phases_per_layer_per_dnn[dnn.name],
                                                                 generate_schedule)
        buffers_per_dnn.append(csdf_buffers)
        if generate_schedule:
            copy_dnn_schedule(dnn.name, dnn_schedule, schedule)

    # reuse buffers among dnn (csdf)
    shared_buffers = reuse_buffers_among_csdf(buffers_per_dnn, [dnn.name for dnn in dnns])
    set_auto_buffer_names(shared_buffers)

    if not generate_schedule:
        schedule = None

    return shared_buffers, schedule


########################################
# multi-dnn with no pipeline parallelism

def get_mms_buffers_multi_pipelined(partitions_per_dnn: [],
                                    phases_per_layer_per_partition_per_dnn: [],
                                    dnn_names: [str],
                                    generate_schedule=False):
    """
    Get buffers with data processing by parts and buffers reuse for a multi-CNN application
        where memory reused within and among dnns and pipeline parallelism is exploited
    :param partitions_per_dnn: list [partitions_1, partitions_2, ..., partitionsN] where
        partitions_i is a list of partitions (sub-networks) of a DNN, N is the total number of DNNs
    :param phases_per_layer_per_partition_per_dnn: list
        [phases_per_partition_1, phases_per_partition_2, ..., phases_per_partition_N] where
        phases_per_partition_j is a dictionary with key = dnn (partition) name, value = dictionary
        with phases (values) per dnn layer (keys)
    :param dnn_names: list of DNN names for DNN partitions
        :param generate_schedule: (flag) if yes, actual per-partition-per dnn schedule is generated. Otherwise,
        null-schedule is returned
    :return: list of CSDF buffers, used by the application, and schedule,
        required for the application to be executed with proposed buffers
    """
    buffers_per_dnn = []

    schedule = MMSDNNInfModelSchedule(dnn_names)

    for dnn_id in range(len(partitions_per_dnn)):
        partitions = partitions_per_dnn[dnn_id]
        dnn_name = dnn_names[dnn_id]
        phases_per_layer_per_partition = phases_per_layer_per_partition_per_dnn[dnn_id]

        # single-partition dnn (executed sequentially)
        if len(partitions) == 1:
            single_partition = partitions[0]
            phases_per_layer = phases_per_layer_per_partition[single_partition.name]
            dnn_buffers, dnn_schedule = get_mms_buffers_no_pipeline(single_partition,
                                                                    phases_per_layer,
                                                                    generate_schedule)

        # multi-partition dnn (executed as a pipeline)
        else:
            dnn_buffers, dnn_schedule = get_mms_buffers_pipelined(partitions,
                                                                  phases_per_layer_per_partition,
                                                                  dnn_names[dnn_id],
                                                                  generate_schedule)
        buffers_per_dnn.append(dnn_buffers)

        if generate_schedule:
            copy_dnn_schedule(dnn_name, dnn_schedule, schedule)

    # reuse buffers among dnns (csdf)
    shared_buffers = reuse_buffers_among_csdf(buffers_per_dnn, dnn_names)
    set_auto_buffer_names(shared_buffers)

    if not generate_schedule:
        schedule = None

    return shared_buffers, schedule

########################################
# helper functions


def annotate_with_sim_time(dnn):
    """
    Annotate layers with fake time to simulate their schedule
    # NOTE: this is NECESSARY for timed simulation!
    """
    for layer in dnn.get_layers():
        layer.time_eval = max(layer.phases, 1)


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


def associate_buffers_with_csdf_model(csdf_buffers: [], csdf_model_name):
    """
    Associate all channels of every CSDF buffer in the input CSDF buffers list with a specific CSDF model
    :param csdf_buffers:  CSDF buffers
    :param csdf_model_name: name of the CSDF model
    """
    for csdf_buffer in csdf_buffers:
        csdf_buffer.csdf_model_name_per_channel = []
        for channel in csdf_buffer.channels:
            csdf_buffer.csdf_model_name_per_channel.append(csdf_model_name)


