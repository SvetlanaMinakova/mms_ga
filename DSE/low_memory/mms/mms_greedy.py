from techniques.buf_reuse import build_reuse_buffers, build_naive_buffers
from eval.memory.dnn_model_mem_eval import eval_connection_buffer_tokens, eval_reused_buffers_mb, eval_reused_buffer_tokens
from techniques.dp_by_parts import get_max_phases


def get_optimal_phases_for_dnns_set(dnns, max_memory_mb, verbose=False, token_size=4):
    """
    Get optimal number of phases per dnn
    :param dnns list of all dnns
    :param max_memory_mb memory constraint in mb
    :param verbose
    :param token_size token size
    :return: optimal number of phases per dnn
    """
    # initialize all phases with 1
    phases_per_dnn = get_phases_per_dnn_init_with_ones(dnns)

    # estimate naive memory for comparison
    naive_buf_memory = eval_buf_memory_naive(dnns, token_size)

    # build reuse buffers
    reuse_buffers = build_reuse_buffers(dnns, reuse_among_dnn=True)
    reuse_buf_memory = eval_reused_buffers_mb(reuse_buffers, token_size)
    mem_reduction_sufficient = (reuse_buf_memory <= max_memory_mb)

    if verbose:
        print("memory after reuse         :", round(reuse_buf_memory, 2),
              "reduced from", round(naive_buf_memory, 2), "MB to", round(reuse_buf_memory, 2), "MB")
        print("memory reduction sufficient: ", mem_reduction_sufficient, "(", max_memory_mb, "MB required)")

    if not mem_reduction_sufficient:
        if verbose:
            print("I am starting to introduce phases")

        # introduce phases until dnns fit in memory
        introduce_phases_until_mem_fits(reuse_buffers, phases_per_dnn, max_memory_mb, token_size)

        reuse_buf_memory = eval_reused_buffers_mb(reuse_buffers, token_size)
        mem_reduction_sufficient = (reuse_buf_memory <= max_memory_mb)

        if verbose:
            print("memory after reuse + dp    :", round(reuse_buf_memory, 2),
                  "reduced from", round(naive_buf_memory, 2), "MB to", round(reuse_buf_memory, 2), "MB")
            print("memory reduction sufficient: ", mem_reduction_sufficient, "(", max_memory_mb, "MB required)")

            print("I am done")

    return phases_per_dnn


def eval_buf_memory_naive(dnns, token_size):
    """
    Eval size of naive buffers for reference
    :param dnns: dnns
    :param token_size size of one token in bytes
    :return: size of naive buffers  built for dnns
    """
    naive_buf = build_naive_buffers(dnns)
    buffers_md = eval_reused_buffers_mb(naive_buf, token_size)
    return buffers_md


def introduce_phases_until_mem_fits(reuse_buffers, phases_per_dnn, max_buf_size, token_size):
    """
    Introduce phases into dnn until memory fits into constraint
    :param reuse_buffers: reuse buffers
    :param phases_per_dnn: dict of structure key: dnn_name (str), value: phases_per_layers ({})
    phases-per-layer dict has structure key: layer_name (str), phases_per_layer (int)
    :param max_buf_size: max buffers size (constraint)
    :param token_size: size of data token
    :return:
    """

    buffers_to_visit = [reuse_buf for reuse_buf in reuse_buffers]

    visited_edges_per_buf = {}
    for buffer in buffers_to_visit:
        visited_edges_per_buf[buffer] = []

    all_buffers_size = eval_reused_buffers_mb(reuse_buffers, token_size)

    while all_buffers_size > max_buf_size and len(buffers_to_visit) > 0:
        largest_buf_to_visit = find_largest_buf(buffers_to_visit)
        largest_edge, dnn_name = find_largest_unvisited_edge_in_buf(largest_buf_to_visit,
                                                                    visited_edges_per_buf[largest_buf_to_visit])

        while largest_edge is None and len(buffers_to_visit) > 0:
            # print("largest edge is none!")
            # remove buffer where all the edges are visited
            buffers_to_visit.remove(largest_buf_to_visit)
            # find next largest buffer
            largest_buf_to_visit = find_largest_buf(buffers_to_visit)
            # find next largest edges
            if len(buffers_to_visit) > 0:
                largest_edge, dnn_name = find_largest_unvisited_edge_in_buf(largest_buf_to_visit,
                                                                            visited_edges_per_buf[largest_buf_to_visit])
        if largest_edge is not None:
            introduce_phases(largest_buf_to_visit, largest_edge, dnn_name, phases_per_dnn)
            visited_edges_per_buf[largest_buf_to_visit].append(largest_edge)
            # print("introduce phases in dnn", dnn_name, "edge", largest_edge)
            # print("visited edges per buf: ", visited_edges_per_buf[largest_buf_to_visit])

        all_buffers_size = eval_reused_buffers_mb(reuse_buffers, token_size)

    # print("I've got buf sizes         :", eval_reused_buffers_size_with_phases(reuse_buffers, phases_per_dnn))
    # print("I found largest edge       : ", largest_edge, "!")
    # print("It belongs to dnn with name: ", dnn_name)
    # print("I've introduced phases into it!")
    # print("Now buf sizes are          :", eval_reused_buffers_size_with_phases(reuse_buffers, phases_per_dnn))


def introduce_phases(buf, edge, dnn_name, phases_per_dnn):
    """
    Introduce phases into an edge
    :param buf buffer, storing the edge
    :param edge: edge
    :param dnn_name: name of dnn, to which the edge belongs
    :param phases_per_dnn: phases per dnn
    """
    max_src_phases = get_max_phases(edge.src)
    max_dst_phases = get_max_phases(edge.dst)

    phases_per_layer = phases_per_dnn[dnn_name]
    phases_per_layer[edge.src.name] = max(phases_per_layer[edge.src.name], max_src_phases)
    phases_per_layer[edge.dst.name] = max(phases_per_layer[edge.dst.name], max_dst_phases)
    edge.src.phases = max_src_phases
    edge.dst.phases = max_dst_phases

    # update buffer size
    buf.size = eval_reused_buffer_tokens(buf)


def find_largest_buf(buffers):
    """
    Find largest buffer in the buffers list
    :param buffers: buffers list
    :return: largest buffer in the buffers list
    """
    if len(buffers) == 0:
        return None

    largest_buf = buffers[0]
    for i in range(1, len(buffers)):
        if largest_buf.size < buffers[i].size:
            largest_buf = buffers[i]
    return largest_buf


def find_largest_unvisited_edge_in_buf(buffer, visited_edges):
    if buffer.is_empty():
        return None

    largest_edge = None
    dnn_name = None
    largest_edge_size = 0.0

    for con_per_dnn in buffer.connections_per_dnn.items():
        cur_dnn, cur_connections = con_per_dnn
        for edge in cur_connections:
            if edge not in visited_edges:
                if eval_connection_buffer_tokens(edge) > largest_edge_size:
                    largest_edge = edge
                    largest_edge_size = eval_connection_buffer_tokens(edge)
                    dnn_name = cur_dnn.name
    return largest_edge, dnn_name


def get_phases_per_dnn_init_with_ones(dnns):
    """
    Create dict of phases per-dnn, per-layer, where every layer performs one phase
    :param dnns: dnns
    :return: dict of phases per-dnn, per-layer, where every layer performs one phase
    """
    phases_per_dnn = {}
    for dnn in dnns:
        dnn_name = dnn.name
        dnn_phases_per_layer = {}
        for layer in dnn.get_layers():
            dnn_phases_per_layer[layer.name] = 1
        phases_per_dnn[dnn_name] = dnn_phases_per_layer
    return phases_per_dnn

