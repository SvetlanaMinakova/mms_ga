from eval.latency.aloha.ops_tensor_transformer import apply_transformation_stages, get_round_ops, get_data_type_tiles
from eval.latency.aloha.ops_tensor_transformer import get_data_t
from util import giga, milli
from eval.latency.aloha.ops_tensor_builder import build_ops_tensor
from models.edge_platform.aloha_eval_platform_spec.platform_spec import EmbeddedPlatform, Processor


def eval_layer_perf_ms(layer, platform: EmbeddedPlatform, processor: Processor, eval_data=True, tile_data=False, verbose=False):
    """
    Eval latency in milliseconds (ms) of a dnn layer, using ALOHA eval
    :param layer: layer
    :param platform: ALOHA platform model
    :param processor: processor of ALOHA platform model
    :param eval_data: (flag): If true, data transfers will be taken into account
    :param tile_data: (flag): If true, data transfers will be tiled with respect to the cache memory
    :param verbose: (flag): If true, details will be printed
    :return: latency in milliseconds (ms) of a dnn layer, estimated using ALOHA eval
    """
    if layer.built_in:
        return 0.0

    ops_t = build_ops_tensor(layer)
    data_t_untiled = get_data_t(layer, ops_t, processor.unrolling, platform.data_bytes)
    ops_t, data_t_tiled = apply_transformation_stages(layer, ops_t, platform, processor, verbose)

    data_t = data_t_untiled
    if tile_data:
        data_t = data_t_tiled
    exec_time_ms = analyze_aloha_ops_and_data_t(layer, ops_t, data_t, platform, processor, eval_data, verbose)

    return exec_time_ms


def analyze_aloha_ops_and_data_t(layer, ops_t, data_t, platform, processor, eval_data=True, verbose=False):
    round_ops = get_round_ops(ops_t)
    ops_exec_time_ms = round_ops/(processor.max_perf * (giga() * milli()))

    tiles = ops_t["tiles"] if "tiles" in ops_t.keys() else 1
    i_data_bytes = 0
    o_data_bytes = 0
    w_data_bytes = 0

    inp_data_transfer_time = 0
    outp_data_transfer_time = 0
    weights_transfer_time = 0

    # eval input_examples for follwing data_types = ["input_data", "output_data", "weights"]
    if eval_data:
        i_data_bytes = bytes_per_typed_data_transfer("input_data", data_t)
        if i_data_bytes > 0:
            i_data_limit = processor.unrolling.find_limit_by_resource_name("input_data")
            i_data_tiles = get_data_type_tiles(layer, ops_t, i_data_limit, processor.unrolling)
            i_data_tiles = i_data_tiles * tiles
            i_data_bytes = i_data_bytes * i_data_tiles

        o_data_bytes = bytes_per_typed_data_transfer("output_data", data_t)
        if o_data_bytes >0:
            o_data_limit = processor.unrolling.find_limit_by_resource_name("output_data")
            o_data_tiles = get_data_type_tiles(layer, ops_t, o_data_limit, processor.unrolling)
            o_data_tiles = o_data_tiles * tiles
            o_data_bytes = o_data_bytes * o_data_tiles

        w_data_bytes = bytes_per_typed_data_transfer("weights", data_t)
        if w_data_bytes >0:
            w_data_limit = processor.unrolling.find_limit_by_resource_name("weights")
            w_data_tiles = get_data_type_tiles(layer, ops_t, w_data_limit, processor.unrolling)
            w_data_tiles = w_data_tiles * tiles
            w_data_bytes = w_data_bytes * w_data_tiles

        i_bw = find_bandwidth_ceiling("input_data", platform, processor)
        o_bw = find_bandwidth_ceiling("output_data", platform, processor)
        w_bw = find_bandwidth_ceiling("weights", platform, processor)

        inp_data_transfer_time = data_transfer_ms(i_data_bytes, i_bw)
        outp_data_transfer_time = data_transfer_ms(o_data_bytes, o_bw)
        weights_transfer_time = data_transfer_ms(w_data_bytes, w_bw)

    exec_time_ms = max(ops_exec_time_ms, inp_data_transfer_time, outp_data_transfer_time, weights_transfer_time)
    bound = "ops" if exec_time_ms == ops_exec_time_ms else "input_examples"
    if verbose:
        print("total input_examples (bytes) transferred/time (ms) spent: ")
        print("  - input_examples  : ", i_data_bytes, "bytes,", inp_data_transfer_time, "ms")
        print("  - weights: ", w_data_bytes, "bytes,", weights_transfer_time, "ms")
        print("  - output : ", o_data_bytes, "bytes,", outp_data_transfer_time, "ms")
        print("total ops executed/time (ms) spent: ")
        print(round_ops, "ops,", ops_exec_time_ms, "ms")
        print("bound:", bound)
        print("exec time:", exec_time_ms, "ms")

    return exec_time_ms


def data_transfer_ms(data_bytes, bandwidth_gbs):
    """
    Compute input_examples transfer time
    :param data_bytes: transferred input_examples (bytes)
    :param bandwidth_gbs: bandwidth (GB/s)
    :return:
    """
    if data_bytes == 0 or bandwidth_gbs == 0:
        return 0
    time_ms = data_bytes/(bandwidth_gbs * (giga() * milli()))
    return time_ms


def bytes_per_typed_data_transfer(data_type, data_t):
    """
    Extract number of bytes per input_examples type (aloha) from input_examples tensor
    :param data_t: input_examples tensor
    :param data_type input_examples type
    :return: bytes per transfer of input_examples of specific type
    """
    for data_transfer in data_t:
        (_, transferred_data_type, data_size, _) = data_transfer
        if transferred_data_type == data_type:
            return data_size
    return 0


def find_bandwidth_ceiling(data_type, platform, processor):
    bw = 0
    try:
        data_mem = processor.data_on_memory_mapping[data_type]
        data_ch = platform.get_channel("main", data_mem)
        if data_ch is None:
            data_mem = processor.data_on_memory_mapping[data_type]
            data_ch = platform.get_channel(data_mem, "main")
        if data_ch is None:
            return 0
        else:
            bw = data_ch.bandwidth
            return bw
    except Exception:
        return 0










