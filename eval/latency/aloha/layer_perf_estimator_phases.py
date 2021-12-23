from eval.latency.aloha.ops_tensor_transformer import apply_transformation_stages, get_round_ops, get_data_type_tiles
from eval.latency.aloha.ops_tensor_transformer import get_data_t
from util import giga, milli
from eval.latency.aloha.ops_tensor_builder import build_ops_tensor
from models.edge_platform.aloha_eval_platform_spec.platform_spec import EmbeddedPlatform, Processor
import copy
from eval.latency.aloha.layer_perf_estimator import analyze_aloha_ops_and_data_t


def eval_layer_perf_ms_phases(layer, platform: EmbeddedPlatform, processor: Processor, eval_data=True, tile_data=False, verbose=False):
    """
    Eval latency in milliseconds (ms) of a dnn layer processing data by parts, using ALOHA eval
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

    def __layer_reuses_inp_data():
        if layer.op not in ["conv", "pool"]:
            return False
        if layer.phases == 1:
            return False
        if layer.stride >= layer.fs:
            return False
        return True

    layer_part = copy.deepcopy(layer)
    layer_part.oh = max(int(layer.oh/layer.phases), 1)
    layer_part.ih = layer.fs if __layer_reuses_inp_data() else max(layer.ih/layer.phases, 1)

    ops_t = build_ops_tensor(layer_part)
    data_t_untiled = get_data_t(layer_part, ops_t, processor.unrolling, platform.data_bytes)
    ops_t, data_t_tiled = apply_transformation_stages(layer_part, ops_t, platform, processor, verbose)

    data_t = data_t_untiled
    if tile_data:
        data_t = data_t_tiled

    phase_exec_time_ms = analyze_aloha_ops_and_data_t(layer, ops_t, data_t, platform, processor, eval_data, verbose)
    layer_exec_time_ms = phase_exec_time_ms * layer.phases
    return layer_exec_time_ms

