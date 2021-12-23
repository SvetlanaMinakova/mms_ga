"""
This module estimates performance (exec. time), energy and memory of a dnn or set of dnns.
The estimation of latency and energy is performed using one of the 
following estimation methods: 'ops', 'lut', 'sroof' or 'aloha'
"""

from techniques.scheduling.pipeline_greedy import map_greedy_pipeline, get_jetson, get_sum_proc_time
from techniques.scheduling.sequential_greedy import map_greedy_sequenatial
from converters.dnn_to_task_graph import dnn_to_task_graph
from eval.throughput.eval_matrix_builder import build_time_eval_matrix


def estimate_dnns_perf_and_thr(dnns, lut_file_per_proc_type: {}, pipeline=False, verbose=True):
    """
    Estimate performance (milliseconds) of a set of dnns
    :param dnns: dnns set of dnns, each represented as the dnn model
    :param lut_file_per_proc_type: dictionary, where key = lut file name,
     value= path to lut file with on-board per-layer execution time measurements for the specified processor type
    :param pipeline: If True, dnn is scheduled as a pipeline.
    Otherwise, dnn is scheduled sequentially
    :param verbose: verbose
    :return: performance of dnns
    """

    # dnn exec times to return
    dnn_times = []
    dnn_throughputs = []

    arch, gpu_id = get_architecture_and_gpu_id("jetson")

    for dnn in dnns:
        dnn_perf = estimate_dnn_perf(dnn, arch, gpu_id, pipeline, lut_file_per_proc_type, verbose)
        dnn_thr = perf_to_thr(dnn_perf)
        dnn_times.append(dnn_perf)
        dnn_throughputs.append(dnn_thr)

    return dnn_times, dnn_throughputs


def estimate_dnn_perf(dnn, architecture, gpu_id, pipeline, lut_file_per_proc_type: {}, verbose=True):
    """
    Estimate performance (n milliseconds) of a dnn
    :param dnn: dnn
    :param lut_file_per_proc_type: dictionary, where key = lut file name,
     value= path to lut file with on-board per-layer execution time measurements for the specified processor type
    :param architecture target platform architecture
    :param gpu_id id of processor-accelerator in the target platform
    :param pipeline: If True, dnn is scheduled as a pipeline.
    :param verbose: verbose
    Otherwise, dnn is scheduled sequentially
    :return: dnn performance in milliseconds
    """

    # convert dnn into task graph
    app_graph = dnn_to_task_graph(dnn)

    # get gpu type id
    gpu_type_id = architecture.get_proc_type_id(gpu_id)

    # get MOC time evaluation matrix, which shows execution time of every dnn layer on every platform processor
    time_eval_matrix = build_time_eval_matrix(dnn, app_graph, architecture, gpu_type_id, lut_file_per_proc_type, verbose)

    if pipeline:
        dnn_perf_ms = estimate_perf_pipeline(app_graph, architecture, time_eval_matrix, gpu_id)
    else:
        dnn_perf_ms = estimate_perf_sequential(app_graph, architecture, time_eval_matrix, gpu_id)

    return dnn_perf_ms


def estimate_perf_pipeline(app_graph, architecture, time_eval_matrix, gpu_id):
    """
    Estimate performance (n milliseconds) of a dnn, executed as a pipeline
    :param app_graph application graph
    :param architecture target platform archotecture
    :param gpu_id id of processor-accelerator in the target platform
    :param time_eval_matrix:
        # matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        # n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    :return: dnn performance in milliseconds
    """
    # build pipeline schedule/mapping
    greedy_pipeline_mapping = map_greedy_pipeline(app_graph, architecture, time_eval_matrix, gpu_id)
    total_time = 0.0
    for proc_id in range(len(architecture.processors)):
        proc_type_id = architecture.get_proc_type_id(proc_id)
        proc_time = get_sum_proc_time(time_eval_matrix, greedy_pipeline_mapping[proc_id], proc_type_id)
        total_time = max(total_time, proc_time)

    return total_time


def estimate_perf_sequential(app_graph, architecture, time_eval_matrix, gpu_id):
    """
    Estimate performance (n milliseconds) of a dnn, executed sequentially
    :param app_graph application graph
    :param architecture target platform archotecture
    :param gpu_id id of processor-accelerator in the target platform
    :param time_eval_matrix:
        # matrix [M]x[N] where m in [0, len (processor_types_distinct)-1] represents processor type,
        # n in [0, layers_num] represents layer, matrix[m][n] contains execution time of layer n on processor m
    :return: dnn performance in milliseconds
    """
    # build sequential schedule/mapping
    greedy_sequential_mapping = map_greedy_sequenatial(app_graph, architecture, time_eval_matrix, gpu_id)

    total_time = 0.0
    for proc_id in range(len(architecture.processors)):
        proc_type_id = architecture.get_proc_type_id(proc_id)
        proc_time = get_sum_proc_time(time_eval_matrix, greedy_sequential_mapping[proc_id], proc_type_id)
        total_time += proc_time

    return total_time


def get_architecture_and_gpu_id(arch_name):
    if arch_name == 'jetson':
        # get example platform: jetson
        architecture = get_jetson()
        gpu_id = 5
        return architecture, gpu_id
    raise Exception("Unknown architecture", arch_name, "only jetson is supported for now")


def perf_to_thr(perf_ms):
    if perf_ms == 0:
        return 0

    ms_in_s = 1000.0
    thr = (1 * ms_in_s)/perf_ms
    return thr