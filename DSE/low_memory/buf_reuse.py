"""
This module contains DNN buffers reuse technique
"""
from eval.memory.dnn_model_mem_eval import eval_connection_buffer_tokens
from models.data_buffers import DNNDataBuffer


def build_reuse_buffers(dnns_list, reuse_among_dnn=True, pipelined_dnns_mappings=None):
    """
    Build a set of reuse buffers for list of dnns
    :param dnns_list: list of dnns
    :param reuse_among_dnn if True, buffers will be reused among dnns
    Otherwise, buffers will only be reused within one dnn
    :param pipelined_dnns_mappings: dictionary where key = dnn name, value= dnn pipelined mapping
    (if None, we assume CNNs are executed as a sequence, i.e., without pipeline)
    :return:  set of reuse buffers for list of dnns
    """
    buffers = []
    for dnn in dnns_list:
        for connection in dnn.get_connections():
            reuse_buffer = find_reuse_buffer(buffers, dnn, connection, reuse_among_dnn, pipelined_dnns_mappings)
            if reuse_buffer is None:
                reuse_buffer = DNNDataBuffer("B" + str(len(buffers)), 0)
                buffers.append(reuse_buffer)
            reuse_buffer.add_connection(dnn, connection)
    return buffers


def find_reuse_buffer(reuse_buffers, dnn, new_connection, reuse_among_dnn, pipelined_dnns_mappings):
    """
    Within a list of reused buffers, find a buffer which can be reused to store data of a connection of a dnn
    :param reuse_buffers:  list of reused buffers
    :param dnn: dnn
    :param new_connection: new connection
    :param reuse_among_dnn fi True, buffers will be reused among dnns
    Otherwise, buffers will only be reused within one dnn
    :param pipelined_dnns_mappings: dictionary where key = dnn name, value= dnn pipelined mapping
    (if None, we assume CNNs are executed as a sequence, i.e., without pipeline)
    :return: a buffer which can be reused to store data of a connection
    of a dnn if such buffer exists and None otherwise
    """
    def reusable_for_connection(reuse_buf):
        # empty buffer is always reusable
        if reuse_buf.is_empty():
            return True

        # among-dnn storage check
        stores_other_dnn = __stores_other_dnn(reuse_buf)
        if reuse_among_dnn is False and stores_other_dnn:
            return False

        # within-dnn storage check
        stores_incompatible = __stores_incompatible_connections_of_same_dnn(reuse_buf)
        if stores_incompatible:
            return False

        # pipeline execution check
        if __stores_incompatible_partitions(reuse_buf):
            return False

        return True

    def __stores_other_dnn(reuse_buf):
        for stored_dnn in reuse_buf.connections_per_dnn.keys():
            if dnn.name != stored_dnn.name:
                return True

    def __stores_dnn(reuse_buf):
        for stored_dnn in reuse_buf.connections_per_dnn.keys():
            if dnn.name == stored_dnn.name:
                return True

    def __stores_incompatible_partitions(reuse_buf):
        if dnn not in reuse_buf.connections_per_dnn.keys():
            return False

        if pipelined_dnns_mappings is None:
            return False

        if dnn.name not in pipelined_dnns_mappings.keys():
            return False

        pipeline_mapping = pipelined_dnns_mappings[dnn.name]
        new_src_proc_id = find_proc_id(new_connection.src, pipeline_mapping)
        # new_dst_proc_id = find_proc_id(new_connection.dst, pipeline_parallelism)
        stored_connections = reuse_buf.connections_per_dnn[dnn]
        for stored_connection in stored_connections:
            stored_src_proc_id = find_proc_id(stored_connection.src, pipeline_mapping)
            # new_dst_proc_id = find_proc_id(new_connection.dst, pipeline_parallelism)
            if stored_src_proc_id != new_src_proc_id:
                return True

        return False

    def __stores_incompatible_connections_of_same_dnn(reuse_buf):
        if dnn not in reuse_buf.connections_per_dnn.keys():
            return False

        stored_connections = reuse_buf.connections_per_dnn[dnn]
        for stored_connection in stored_connections:
            # I/O of the same layer
            if stored_connection.src.id == new_connection.dst.id:
                return True
            if stored_connection.dst.id == new_connection.src.id:
                return True
            # residual connections
            if stored_connection.src.id <= new_connection.src.id and stored_connection.dst.id >= new_connection.dst.id:
                return True
            if new_connection.src.id <= stored_connection.src.id and new_connection.dst.id >= stored_connection.dst.id:
                return True

    # main script
    # TODO: extend with best-fitting here
    new_connection_tokens = eval_connection_buffer_tokens(new_connection)
    reusable_buffers = []
    for buffer in reuse_buffers:
        if reusable_for_connection(buffer):
            reusable_buffers.append(buffer)

    best_reusable_buffer = find_best_reusable_buffer(reusable_buffers, new_connection_tokens)
    return best_reusable_buffer


def find_best_reusable_buffer(reusable_buffers, new_connection_tokens):
    if not reusable_buffers:
        return None
    # cost for storing this connection in platform memory
    cost = new_connection_tokens
    best_buffer_id = -1
    for buffer_id in range(len(reusable_buffers)):
        buffer = reusable_buffers[buffer_id]
        buf_cost = max(new_connection_tokens-buffer.size, 0)
        if buf_cost < cost:
            best_buffer_id = buffer_id
            cost = buf_cost

    # no suitable buffer was found
    if best_buffer_id == -1:
        return None
    # suitable buffer was found
    return reusable_buffers[best_buffer_id]


def get_top_n_memory_users(reused_buffers: [DNNDataBuffer], n=3):
    """
    Get top n memory-heavy users (dnn + connection)
    :param reused_buffers: list of reuse buffers
    :param n: n
    :return: top n memory-heavy users (dnn + connection)
    """
    mem_user_connections = []
    mem_user_dnns = []
    mem_user_buffers = []

    for i in range(n):
        dnn_user = None
        connection_user = None
        buf_id_user = None
        max_tokens = -1

        buffer_id = 0
        for b in reused_buffers:
            for dnn in b.connections_per_dnn.keys():
                for connection in b.connections_per_dnn[dnn]:
                    if connection not in mem_user_connections:
                        connection_tokens = eval_connection_buffer_tokens(connection)
                        if connection_tokens > max_tokens:
                            buf_id_user = buffer_id
                            dnn_user = dnn
                            connection_user = connection
                            max_tokens = connection_tokens
            buffer_id += 1
        mem_user_buffers.append(buf_id_user)
        mem_user_dnns.append(dnn_user)
        mem_user_connections.append(connection_user)

    zipped_results = []
    for i in range(len(mem_user_connections)):
        zipped_results.append((mem_user_buffers[i], mem_user_dnns[i], mem_user_connections[i]))

    return zipped_results


def print_top_n_memory_users(reused_buffers: [DNNDataBuffer], n=3):
    """
    Print top n memory-heavy users (dnn + connection)
    :param reused_buffers: list of reuse buffers
    :param n: n
    :return: top n memory-heavy users (dnn + connection)
    """
    top_n_memory_users = get_top_n_memory_users(reused_buffers)

    for i in range(n):
        user = top_n_memory_users[i]
        buf_id, dnn_user, connection_user = user
        dnn_name = None if dnn_user is None else dnn_user.name
        connection_size = eval_connection_buffer_tokens(connection_user) if connection_user is not None else 0
        connection_str = "None"
        if connection_user is not None:
            connection_str = "{" + str(connection_user.src.id) + " (" + connection_user.src.op + ") --> "
            connection_str += "{" + str(connection_user.dst.id) + " (" + connection_user.dst.op + ")}"

            print("  buffer_id:", buf_id, "; dnn: ", dnn_name,
                  "; connection: ", connection_str, "; size: ", connection_size)


def annotate_with_double_buffers(dnn, pipeline_schedule):
    for connection in dnn.get_connections():
        rdb = requires_double_buffer(connection, pipeline_schedule)
        connection.double_buffer = rdb


def requires_double_buffer(connection, pipeline_schedule):
    src_id = connection.src.id
    dst_id = connection.dst.id
    scr_proc_id = find_task_processor(src_id, pipeline_schedule)
    dst_proc_id = find_task_processor(dst_id, pipeline_schedule)
    if scr_proc_id != dst_proc_id:
        return True
    return False


def find_task_processor(task_id, schedule: []):
    for proc_id in range(len(schedule)):
        proc_tasks = schedule[proc_id]
        if task_id in proc_tasks:
            return task_id
    return -1


def reset_double_buffers(dnn):
    for connection in dnn.get_connections():
        connection.double_buffer = False


def set_all_double_buffers(dnn):
    for connection in dnn.get_connections():
        connection.double_buffer = True


def set_double_buffers_from_pipeline_mapping(dnn, pipeline_mapping):
    for connection in dnn.get_connections():
        src_proc_id = find_proc_id(connection.src, pipeline_mapping)
        dst_proc_id = find_proc_id(connection.dst, pipeline_mapping)
        if src_proc_id != dst_proc_id:
            connection.double_buffer = True
            # print("double buffer set for connection ", str(connection))
        else:
            connection.double_buffer = False


def find_proc_id(layer, pipeline_mapping):
    for proc_id in range(len(pipeline_mapping)):
        proc_tasks = pipeline_mapping[proc_id]
        if layer.id in proc_tasks:
            return proc_id
    return -1