from util import mega, elements_prod
from models.dnn_model.param_shapes_generator import generate_param_shapes_dict
"""
This module evaluates dnn_model memory (in MB), taking into account DNN schedule and topology
"""


def eval_dnn_memory(dnn, bytes_per_pixel):
    """
    Eval dnn_model memory
    :param dnn: dnn_model
    :param bytes_per_pixel: bytes per wights/data pixel
    :return: DNN memory in megabytes (MB)
    """
    weights_tokens = dnn_weights_tokens(dnn)
    buffer_tokens = eval_dnn_buffer_tokens(dnn)
    memory = (weights_tokens + buffer_tokens)/float(mega()) * bytes_per_pixel
    return memory


def eval_dnn_list_memory(dnns, bytes_per_pixel):
    """
    Eval memory of several dnns
    :param dnns: list of dnns
    :param bytes_per_pixel: bytes per wights/data pixel
    :return: DNN memory in megabytes (MB)
    """
    memory = []
    for dnn in dnns:
        dnn_memory = eval_dnn_memory(dnn, bytes_per_pixel)
        memory.append(dnn_memory)
    return memory


# ******** #
#  weights #

def eval_dnn_weights(dnn, token_size=4):
    """
    Eval dnn weights size in MB
    :param dnn: dnn model
    :param token_size: bytes per wights/data pixel
    :return: dnn weights size in MB
    """
    weights_tokens = dnn_weights_tokens(dnn)
    weights_mb = (weights_tokens*token_size)/1000000.0
    return weights_mb


def dnn_weights_tokens(dnn):
    """
    Get number of elements (tokens) in weights of a DNN
    :param dnn: dnn_model
    :return: number of elements (tokens) in weights of a DNN
    """
    weights = 0
    for layer in dnn.get_layers():
        weights += layer_weights_tokens(layer)
    return weights


def layer_weights_tokens(layer):
    """
    Get number of elements (tokens) in weights of a layer
    :param layer: layer
    :return: number of elements (tokens) in weights of a layer
    """
    weights = 0
    w_shapes = generate_param_shapes_dict(layer)
    for shape in w_shapes.values():
        w_tokens = elements_prod(shape)
        weights += w_tokens
    return weights

# ******* #
# buffers #


def eval_dnn_buffer_tokens(dnn):
    """
    Get number of elements (tokens) in buffers of a DNN
    :param dnn: dnn_model
    :return: number of elements (tokens) in buffers of a DNN
    """
    tokens = 0
    for connection in dnn.get_connections():
        tokens += eval_connection_buffer_tokens(connection)
    return tokens


def eval_reused_buffers_mb(reused_buffers: [], token_size_bytes):
    """
    Eval size of seceral reuse buffers in megabytes
    :param reused_buffers: list of reused buffers
    :param token_size_bytes: size of one token in bytes
    :return: size of reuse buffer (in tokens)
    """
    buffers_mb = 0
    for reused_buffer in reused_buffers:
        buf_mb = eval_reused_buffer_mb(reused_buffer, token_size_bytes)
        buffers_mb += buf_mb
    return buffers_mb


def eval_reused_buffers_tokens(reused_buffers: []):
    """
    Eval number of tokens in several reuse buffers
    :param reused_buffers: list of reused buffers
    :return: size of reuse buffer (in tokens)
    """
    buffers_tokens = 0
    for reused_buffer in reused_buffers:
        buf_tokens = eval_reused_buffer_tokens(reused_buffer)
        buffers_tokens += buf_tokens

    return buffers_tokens


def eval_reused_buffer_mb(reused_buffer, token_size_bytes):
    """
    Eval size of reuse buffer in megabytes
    :param reused_buffer: reused buffer
    :param token_size_bytes: size of one token in bytes
    :return: size of reuse buffer (in tokens)
    """
    buf_tokens = eval_reused_buffer_tokens(reused_buffer)
    buf_mb = tokens_to_mb(buf_tokens, token_size_bytes)
    return buf_mb


def eval_reused_buffer_tokens(reused_buffer):
    """
    Eval number of tokens in the reuse buffer
    :param reused_buffer: reused buffer
    :return: size of reuse buffer (in tokens)
    """
    buf_tokens = 0
    for item in reused_buffer.connections_per_dnn.items():
        dnn, connections_per_dnn = item
        for connection in connections_per_dnn:
            # print(connection)
            connection_tokens = eval_connection_buffer_tokens(connection)
            buf_tokens = max(connection_tokens, buf_tokens)
    return buf_tokens


def eval_connection_buffer_tokens(connection):
    """
    TODO change for multi-input layers!
    Get number of elements (tokens) in a buffer of a DNN connection
    :param dnn: dnn_model
    :param connection: dnn_model connection
    :return: number of elements (tokens) in a buffer of a DNN connection
    """
    if connection.dst.built_in or connection.dst.op == "skip":
        buffer_tokens = 0
        return buffer_tokens

    src = connection.src
    dst = connection.dst

    src_tokens = (src.ofm * src.oh * src.ow) / src.phases if not src.built_in else 0
    dst_tokens = (dst.ifm * dst.ih * dst.iw) / dst.phases
    # TODO: requires upd with operators (make op-independent)
    if dst.op in ["arithmetic", "concat"]:
        dst_tokens = src_tokens

    # conv and pool layers
    if dst.phases > 1 and dst.fs > 1:
        dst_tokens = (dst.ifm * dst.fs * dst.iw)
    buffer_tokens = max(src_tokens, dst_tokens)

    if connection.double_buffer:
        buffer_tokens *= 2

    return buffer_tokens

# ******* #
#  other  #


def tokens_to_mb(tokens, token_size_bytes):
    """
    Convert tokens to megabytes
    :param tokens: tokens
    :param token_size_bytes: size of one token in bytes
    """
    mb = (tokens * token_size_bytes)/ float(mega())
    return mb

