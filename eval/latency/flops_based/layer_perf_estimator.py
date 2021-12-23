from eval.flops.layer_flops_estimator import eval_layer_flops
from util import giga, milli


def eval_layer_latency_sec(layer, gops_per_sec):
    layer_flops = eval_layer_flops(layer)
    layer_sec = layer_flops/(gops_per_sec * float(giga()))
    return layer_sec


def eval_layer_latency_ms(layer, gops_per_sec):
    layer_flops = eval_layer_flops(layer)
    layer_sec = layer_flops/(gops_per_sec * float(giga()))
    layer_ms = layer_sec/(float(milli()))
    return layer_ms

