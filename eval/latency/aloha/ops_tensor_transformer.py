from util import elements_prod
from util import round_div

"""
This module applied 4-steps ALOHA transformation to the CNN layer ops tensor
"""


def apply_transformation_stages(layer, ops_t, platform, processor, verbose=True):
    # loops reordering
    ops_t = set_loop_order(ops_t, processor.unrolling.loops_order)

    if verbose:
        print()
        print("STAGE 1: loops reordering")
        print("ops_t, reordered: ")
        # print(ops_t)
        print_as_nested_loops(ops_t)

    ops_t = unroll(ops_t, processor.parallel_comp_matrix, processor.unrolling)

    if verbose:
        print()
        print("STAGE 2: loops unrolling")
        print("ops_t, unrolled: ")
        print_as_nested_loops(ops_t)
        # print(ops_t)

    round_ops = get_round_ops(ops_t)
    # if verbose:
    #    print("round ops: ", round_ops)

    data_t = get_data_t(layer, ops_t, processor.unrolling, platform.data_bytes)
    # if verbose:
    #    print("data_t, original: ")
    #    print(data_t)

    if verbose:
        print()
        print("STAGE 3: input_examples transfer insertion")
        print("ops_t, with input_examples: ")
        # print(ops_t)
        print_as_nested_loops_with_data(ops_t, data_t)

    ops_t = tile(layer, ops_t, platform, processor, verbose=False)
    tiles = ops_t["tiles"] if "tiles" in ops_t.keys() else 1
    data_t = get_data_t(layer, ops_t, processor.unrolling, platform.data_bytes)

    if verbose:
        print()
        print("STAGE 4: limitations posing (tiling)")
        print("ops_t, tiled: ")
        # print(ops_t)
        print_as_nested_loops_with_data(ops_t, data_t)

    #

    # if verbose:
    #    print("data_t tile (one of", tiles, "tiles")
    #    print(data_t)

    return ops_t, data_t #round_ops


def set_loop_order(ops_t, loops_order):
    """
    Change order of layer computational tensor loops
    :param ops_t: layer computational tensor
    :param loops_order: new loops order
    :return: layer computational tensor with reordered loops
    """
    if len(ops_t.keys()) != len(loops_order):
        print("loop order setting error: ops tensor dimensionality " + str(len(ops_t.keys())) +
              " does not match length of loops order list " + str(len(loops_order)))
        return ops_t

    ops_t_reordered = {}
    ops_t_keys_list = [key for key in ops_t.keys()]

    for pos in loops_order:
        pos_key = ops_t_keys_list[pos]
        ops_t_reordered[pos_key] = ops_t[pos_key]

    return ops_t_reordered


def get_data_t(layer, ops_t, unrolling, bytes_per_pixel):
    """
    Get input_examples tensor: list, where every element describes a input_examples load/store with
    input_examples load/store loop, input_examples name (i/o/weights), input_examples size (in tokens), load/store flag
    :param layer: CNN layer
    :param ops_t: ops tensor
    :param bytes_per_pixel bytes per pixel
    :param unrolling: unrolling of ops_t over parallel matrix of processor
    :return: input_examples tensor
    """
    data_limits = ["input_data", "output_data", "weights"]
    data_t = []
    for limit in unrolling.limits:
        if limit.resource_user_name in data_limits:
            data_load_dim, data_size = get_data_load(layer, ops_t, limit, unrolling, bytes_per_pixel)
            data_t.append((data_load_dim, limit.resource_user_name, data_size, limit.load))
    return data_t


def get_data_load(layer, ops_t, data_limit, unrolling, bytes_per_pixel):
    """
    Get amount of input_examples per load
    :param layer: CNN layer
    :param ops_t: ops tensor
    :param data_limit: input_examples limit
    :param unrolling: layer unrolling over platform processors
    :param bytes_per_pixel: bytes per input_examples pixel
    :return:
    """
    data_load_dim_name = data_limit.load_dim
    dim_names = layer.get_dim_names(data_limit.resource_user_name)
    if not dim_names:
        return data_load_dim_name, 0

    data_load_size = 1
    data_load_dim_met = False
    for dim_name in ops_t.keys():
        if dim_name == data_load_dim_name:
            data_load_dim_met = True
        if data_load_dim_met:
            if dim_name in dim_names: # data_limit.dims:
                # get un-unrolled (sequential) dims into account
                data_load_size = data_load_size * ops_t[dim_name]
            else:
                if "parallel" in dim_name:
                    unrolling_line_id_str = dim_name.replace("parallel", "")
                    unrolling_line_id = int(unrolling_line_id_str)
                    line = unrolling.comp_unrolling[unrolling_line_id]
                    # get unrolled (parallel) dims into account
                    is_unrolling_of_data_dim = False
                    for unrolled_dim_name in line[0]:
                        if unrolled_dim_name in dim_names: # data_limit.dims:
                            # if this specific dim (or an underlying dim) was unrolled:
                            par_dim_id = find_ops_t_dim_id(ops_t, unrolled_dim_name)
                            data_dim_id = par_dim_id - 1
                            ops_t_sorted_dim_list = [key for key in ops_t.keys()]
                            dim_unrolled_by_par = ops_t_sorted_dim_list[data_dim_id]
                            if dim_unrolled_by_par in dim_names:
                                is_unrolling_of_data_dim = True
                    if is_unrolling_of_data_dim:
                        data_load_size = data_load_size * ops_t[dim_name]

    data_load_size = data_load_size * bytes_per_pixel
    return data_load_dim_name, data_load_size


def get_data_type_tiles(layer, ops_t, data_limit, unrolling):
    """
    Get input_examples tiles (per input_examples type)
    :param layer CNN layer
    :param ops_t: ops tensor
    :param data_limit: input_examples limit
    :param unrolling: tensor unrolling
    :return: input_examples tiles (int) for a specific input_examples type
    """
    data_load_dim_name = data_limit.load_dim
    tiles = 1
    dim_names = layer.get_dim_names(data_limit.resource_user_name)
    if not dim_names:
        return tiles

    data_load_dim_met = False
    for dim_name in ops_t.keys():
        if dim_name == data_load_dim_name:
            data_load_dim_met = True
        if not data_load_dim_met:
            if dim_name in dim_names: # data_limit.dims:
                # get un-unrolled (sequential) dims into account
                tiles = tiles * ops_t[dim_name]
            else:
                if "parallel" in dim_name:
                    unrolling_line_id_str = dim_name.replace("parallel", "")
                    unrolling_line_id = int(unrolling_line_id_str)
                    line = unrolling.comp_unrolling[unrolling_line_id]
                    # get unrolled (parallel) dims into account
                    is_unrolling_of_data_dim = False
                    for unrolled_dim_name in line[0]:
                        if unrolled_dim_name in dim_names: # data_limit.dims:
                            # if this specific dim (or an underlying dim) was unrolled:
                            par_dim_id = find_ops_t_dim_id(ops_t, unrolled_dim_name)
                            data_dim_id = par_dim_id - 1
                            ops_t_sorted_dim_list = [key for key in ops_t.keys()]
                            dim_unrolled_by_par = ops_t_sorted_dim_list[data_dim_id]
                            if dim_unrolled_by_par in dim_names:
                                is_unrolling_of_data_dim = True
                    if is_unrolling_of_data_dim:
                        tiles = tiles * ops_t[dim_name]

    return tiles


def unroll(ops_t, parallel_comp_matrix, unrolling) -> {}:
    """
    Unroll computations within blocks
    :param ops_t: ops tensor
    :param parallel_comp_matrix: parallel comp. matrix to unroll over
    :param unrolling unrolling of tensor over parallel comp. matrix
    :return: ops tensor with unrolled computations
    """
    unrolled_ops_t = ops_t
    line_id =0
    for line in unrolling.comp_unrolling:
        unrolled_ops_t = unroll_line(unrolled_ops_t, parallel_comp_matrix, line, line_id)
        line_id = line_id + 1
    return unrolled_ops_t


def unroll_line(ops_t, parallel_comp_matrix, unrolling_line, line_id) -> {}:
    """
    Unroll computations within blocks
    :param ops_t: ops tensor
    :param parallel_comp_matrix: parallel comp. matrix to unroll over
    :param unrolling_line line with unrolling of tensor over a parallel comp. matrix dimension
    :param line_id id of unrolling line
    :return: ops tensor with unrolled computations
    """

    ops_t_dim_names = [key for key in ops_t.keys()]
    ops_t_dim_names_reversed = [ops_t_dim_names[len(ops_t_dim_names) - i -1] for i in range(len(ops_t_dim_names))]

    unrolled_dims = unrolling_line[0]
    comp_matrix_dims = unrolling_line[1]

    unrolled_ops_t_reversed = {}

    for ops_t_dim_name in ops_t_dim_names_reversed:
        ops_t_dim_size = ops_t[ops_t_dim_name]

        # dimension is not unrolled (it's size is preserved)
        if ops_t_dim_name not in unrolled_dims:
            unrolled_ops_t_reversed[ops_t_dim_name] = ops_t_dim_size

        # dimension is unrolled
        else:
            # direct mapping: one op. tensor dim on one block dim
            if len(unrolled_dims) == 1 and len(comp_matrix_dims) == 1:
                # add block dimension
                comp_m_dim_id = comp_matrix_dims[0]
                comp_m_dim_size = parallel_comp_matrix[comp_m_dim_id]
                unroll_direct(comp_m_dim_size, line_id, ops_t, unrolled_ops_t_reversed, ops_t_dim_name)

            # shared mapping: multiple op. tensor dims to one block dim
            if len(unrolled_dims) > 1 and len(comp_matrix_dims) == 1:
                unroll_shared(unrolling_line, parallel_comp_matrix, unrolled_ops_t_reversed, ops_t)

            # distributed mapping: one op. tensor dim to multiple block dims
            if len(unrolled_dims) == 1 and len(comp_matrix_dims) > 1:
                total_comp_matrix_dims_size = elements_prod(comp_matrix_dims)
                unroll_direct(total_comp_matrix_dims_size, line_id, ops_t, unrolled_ops_t_reversed, ops_t_dim_name)

    # reverse tensor back to normal order
    unrolled_ops_t_reversed_keys = [key for key in unrolled_ops_t_reversed.keys()]
    unrolled_ops_t_keys = [unrolled_ops_t_reversed_keys[len(unrolled_ops_t_reversed_keys) - i - 1] for i in range(len(unrolled_ops_t_reversed_keys))]
    unrolled_ops_t = {}
    for key in unrolled_ops_t_keys:
        unrolled_ops_t[key] = unrolled_ops_t_reversed[key]

    return unrolled_ops_t


def unroll_direct(comp_m_dim_size, line_id, ops_t, unrolled_ops_t_reversed, ops_t_dim_name):
    """
    Apply direct unrolling, where one op. tensor dim is unrolled over one processor comp.matrix dim
    """
    # add parallel ops_t dimension
    parr_dim_name = "parallel" + str(line_id)
    parr_dim_size = comp_m_dim_size
    unrolled_ops_t_reversed[parr_dim_name] = parr_dim_size

    # reduce (sequential) ops_t dimension
    ops_t_reduced_dim_size = round_div(ops_t[ops_t_dim_name], parr_dim_size)
    unrolled_ops_t_reversed[ops_t_dim_name] = ops_t_reduced_dim_size


def unroll_shared(unrolling_line, parallel_comp_matrix, unrolled_ops_t_reversed, ops_t):
    unrolled_dims = unrolling_line[0]
    comp_matrix_dims = unrolling_line[1]

    comp_m_dim_id = comp_matrix_dims[0]
    comp_m_dim_size = parallel_comp_matrix[comp_m_dim_id]

    # add parallel ops_t dimension
    parr_dim_name = "parallel" + str(comp_m_dim_id)
    parr_dim_size = comp_m_dim_size
    unrolled_ops_t_reversed[parr_dim_name] = parr_dim_size

    # reduce (sequential) ops_t dimension
    thread_dims_to_unroll = {}
    for key in ops_t.keys():
        if key in unrolled_dims:
            thread_dims_to_unroll[key] = ops_t[key]

    threads_to_unroll = elements_prod(thread_dims_to_unroll.values())
    total_seq_grid_size = round_div(threads_to_unroll, comp_m_dim_size)

    # choose which dims can be unrolled with available comp. resources

    left_sequential = []

    while total_seq_grid_size > comp_m_dim_size and len(thread_dims_to_unroll.keys()) > 1:
        keys_in_order = [key for key in thread_dims_to_unroll.keys()]
        head_key = keys_in_order[0]
        left_sequential.append(head_key)
        del (thread_dims_to_unroll[head_key])

    # add reduced dims
    threads_to_unroll = elements_prod(thread_dims_to_unroll.values())
    total_seq_grid_size = round_div(threads_to_unroll, comp_m_dim_size)
    keys_in_order = [key for key in thread_dims_to_unroll.keys()]

    # find head (top level) unrolled dim
    head_key = keys_in_order[0]

    # replace non-head (lower than top level) unrolled dims with one-sized dims
    for key_id in range(1, (len(keys_in_order))):
        unrolled_ops_t_reversed[keys_in_order[key_id]] = 1

    # replace head (top level) unrolled dim with reduced top level dim
    unrolled_ops_t_reversed[head_key] = total_seq_grid_size

    # add those dimensions, that were left sequential
    for seq in left_sequential:
        unrolled_ops_t_reversed[seq] = ops_t[seq]


def get_round_ops(comp_tensor):
    """
    Get number of rounded operations
    :param comp_tensor: computations tensor
    """
    round_ops = elements_prod(comp_tensor.values())
    return round_ops


def get_round_data_transfers_per_tile(data_tensor, processor):
    """
    Get number of rounded operations
    :param processor processor, performing input_examples transfer
    :param data_tensor: computations tensor
    """
    data_transfers = []

    for data_load_line in data_tensor:
        (data_load_dim, data_name, data_size, load) = data_load_line
        memory_name = processor.data_on_memory_mapping[data_name]
        data_size_bytes = data_size
        data_transfers.append((data_load_dim, data_name, data_size_bytes, load, memory_name))

    return data_transfers


def resource_use_and_limits(round_data_transfers, platform):
    """
    Check, if any of the resource limits are violated during the layer execution
    :param round_data_transfers: rounded-up input_examples transfers
    :param platform: platform
    :return: list of (resource_usage, resource_limit) pairs
    """
    r_use_and_limits = []
    # check memory limits
    memories_occupied = {}
    for data_transfer in round_data_transfers:
        (data_load_dim, data_name, data_size, load, mem_name) = data_transfer
        if mem_name in memories_occupied:
            mem_occupied_upd = memories_occupied[mem_name] + data_size
            memories_occupied[mem_name] = mem_occupied_upd
        else:
            memories_occupied[mem_name] = data_size

    for mem_name in memories_occupied.keys():
        memory = platform.get_memory(mem_name)
        memory_limit = memory.size
        users = []
        for data_transfer in round_data_transfers:
            (data_load_dim, data_name, data_size, load, d_mem_name) = data_transfer
            if mem_name == d_mem_name:
                users.append(data_name)

        r_use_and_limits_line = {
            "name": mem_name,
            "occupancy": memories_occupied[mem_name],
            "limit": memory_limit,
            "occupiers": users
        }
        r_use_and_limits.append(r_use_and_limits_line)

    return r_use_and_limits


def check_resource_limits_violation(r_use_and_limits):
    """
    Check if resource use is violated
    :return: array, where every i-th element is a booleand, specifying
    if i-th resource use violates the resource constraints
    """
    violation = []
    for r_use_and_limits_line in r_use_and_limits:
        r_violation = r_use_and_limits_line['occupancy'] > r_use_and_limits_line['limit']
        violation.append(r_violation)

    return violation


def tile(layer, ops_t, platform, processor, verbose=False):
    tiling_analysis_results = tiling_analysis(layer, ops_t, platform, processor, verbose)
    tiling_is_needed = tiling_analysis_results["continue_tiling"]
    tiling_iteration = 0
    while tiling_is_needed:
        # best_tile_candidate = find_best_tile_candidate(ops_t, tile_candidates)
        best_tile_candidate = tiling_analysis_results["tiling_candidates"][0]
        # print("tiling iteration: ", tiling_iteration)
        if verbose:
            print("tiling iteration: ", tiling_iteration, "tiling dim: ",
                  best_tile_candidate["split_dim"], "of size", best_tile_candidate["dim_size"])

        ops_t = tile_by_two_iteration(ops_t, best_tile_candidate) #tile_iteration(ops_t, best_tile_candidate)
        tiling_analysis_results = tiling_analysis(layer, ops_t, platform, processor, verbose)
        tiling_is_needed = tiling_analysis_results["continue_tiling"]
        tiling_iteration = tiling_iteration + 1
    return ops_t


def tiling_analysis(layer, ops_t, platform, processor, verbose=False):
    data_t = get_data_t(layer, ops_t, processor.unrolling, platform.data_bytes)
    round_data_transfers = get_round_data_transfers_per_tile(data_t, processor)
    r_use_and_limits = resource_use_and_limits(round_data_transfers, platform)
    r_use_and_limits_violation = check_resource_limits_violation(r_use_and_limits)
    tiling_candidates = get_tile_candidates(ops_t, r_use_and_limits, processor.unrolling)
    continue_tiling = True in r_use_and_limits_violation and len(tiling_candidates) > 0

    tiling_analysis_result = {
        "data_t": data_t,
        "round_data_transfers": round_data_transfers,
        "r_use_and_limits": r_use_and_limits,
        "continue_tiling": continue_tiling,
        "tiling_candidates": tiling_candidates
    }

    if verbose:
        print("ops tensor: ", ops_t)
        print("input_examples tensor: ", data_t)
        # print_as_nested_loops(data_t)
        # print_as_nested_loops_with_data(ops_t, data_t)
        print("round input_examples transfers per tile: ", round_data_transfers)
        print("resource use and limits: ", r_use_and_limits)
        print("resource limits violation: ", r_use_and_limits_violation)
        print("continue_tiling: ", continue_tiling)

    return tiling_analysis_result


def get_tile_candidates(ops_t, r_use_and_limits, unrolling):
    tile_candidates = []
    for r_use_and_limits_line in r_use_and_limits:
        if r_use_and_limits_line['occupancy'] > r_use_and_limits_line['limit']:
            resource_tile_candidates = get_resource_tile_candidates(ops_t, r_use_and_limits_line, unrolling)
            for candidate in resource_tile_candidates:
                tile_candidates.append(candidate)
    return tile_candidates


def get_resource_tile_candidates(ops_t, r_use_and_limits_line, unrolling):
    tile_candidates = []
    for user in r_use_and_limits_line["occupiers"]:
        resource = user
        limit_line = unrolling.find_limit_by_resource_name(resource)
        split_dims = limit_line.split_dims
        for split_dim in split_dims:
            if ops_t[split_dim] > 1:
                tile_candidate = {
                    "resource": resource,
                    "limit": r_use_and_limits_line["limit"],
                    "occupancy": r_use_and_limits_line["occupancy"],
                    "split_dim": split_dim,
                    "dim_size": ops_t[split_dim]
                }
                tile_candidates.append(tile_candidate)
    return tile_candidates


def tile_iteration(ops_t, tile_candidate):
    tiles = round_div(tile_candidate["occupancy"], tile_candidate["limit"])
    dim_size = ops_t[tile_candidate["split_dim"]]
    tiled_dim_size = round_div(dim_size, tiles)
    ops_t[tile_candidate["split_dim"]] = tiled_dim_size
    ops_t_with_tiles = add_tiles_to_ops_t(ops_t, tiles)
    return ops_t_with_tiles


# always tile by 2
def tile_by_two_iteration(ops_t, tile_candidate):
    tiles = 2
    dim_size = ops_t[tile_candidate["split_dim"]]
    tiled_dim_size = round_div(dim_size, tiles)
    ops_t[tile_candidate["split_dim"]] = tiled_dim_size
    ops_t_with_tiles = add_tiles_to_ops_t(ops_t, tiles)
    return ops_t_with_tiles



def add_tiles_to_ops_t(ops_t, tiles):
    ops_t_with_tiles = {}
    total_tiles = tiles
    if "tiles" in ops_t.keys():
        total_tiles = total_tiles * ops_t["tiles"]
        # print("total_tiles: ", total_tiles)
    ops_t_with_tiles["tiles"] = total_tiles
    for old_dim in ops_t.keys():
        if old_dim!= "tiles":
            ops_t_with_tiles[old_dim] = ops_t[old_dim]
    return ops_t_with_tiles


def find_ops_t_dim_id(ops_t, name):
    dim_id = 0
    for dim_name in ops_t.keys():
        if dim_name == name:
            return dim_id
        dim_id = dim_id + 1

    return -1


###########################################
#              print functions           #

def print_as_nested_loops(comp_tensor):
    """
    Print computations tensor as nested loops
    :param comp_tensor: computations tensor
    """
    prefix_inc = "  "
    prefix = ""
    for named_dim in comp_tensor.items():
        name = named_dim[0]
        end = named_dim[1]
        print(prefix + "for (" + name + "=0; " + name + "<" + str(end) + "; " + name + "++){")
        prefix = prefix + prefix_inc

    print(prefix + "do MAC;")

    for _ in comp_tensor.items():
        print(prefix + "}")
        prefix = prefix[:(len(prefix)-len(prefix_inc))]


def print_as_nested_loops_with_data(comp_tensor, data_tensor):
    """
    Print computations tensor as nested loops with input_examples load/stores
    :param comp_tensor: computations tensor
    :param data_tensor input_examples tensor (input_examples loads/stores)
    """
    data_load_dims = [data_load_line[0] for data_load_line in data_tensor]

    prefix_inc = "  "
    prefix = ""
    for named_dim in comp_tensor.items():
        name = named_dim[0]
        end = named_dim[1]
        if name in data_load_dims:
            for data_load_line in data_tensor:
                (data_load_dim, data_name, data_size, load) = data_load_line
                if data_load_dim == name:
                    if load:
                        print(prefix + "load(" + str(data_size) + ", " + data_name + ")")

        print(prefix + "for (" + name + "=0; " + name + "<" + str(end) + "; " + name + "++)")
        prefix = prefix + prefix_inc

    print(prefix + "do MAC;")

    dim_names = [key for key in comp_tensor.keys()]
    dim_names_reversed = [dim_names[len(dim_names)-i] for i in range(1, len(dim_names))]

    for dim_name in dim_names_reversed:
        if dim_name in data_load_dims:
            for data_load_line in data_tensor:
                (data_load_dim, data_name, data_size, load) = data_load_line
                if data_load_dim == dim_name:
                    if not load:
                        print(prefix + "store(" + str(data_size) + ", " + data_name + ")")

        print(prefix + "}")
        prefix = prefix[:(len(prefix)-len(prefix_inc))]


def print_as_nested_loops_short(comp_tensor):
    """
    Print computations tensor as nested loops
    :param comp_tensor: computations tensor
    """
    prefix_inc = " "
    prefix = ""
    for named_dim in comp_tensor.items():
        name = named_dim[0]
        end = named_dim[1]
        print(prefix + "for (" + name + "=0; " + name + "<" + str(end) + "; " + name + "++)")
        prefix = prefix + prefix_inc


def print_as_nested_loops_with_data_short(comp_tensor, data_tensor):
    """
    Print computations tensor as nested loops with input_examples load/stores
    :param comp_tensor: computations tensor
    :param data_tensor input_examples tensor (input_examples loads/stores)
    """
    data_load_dims = [data_load_line[0] for data_load_line in data_tensor]

    prefix_inc = " "
    prefix = ""
    for named_dim in comp_tensor.items():
        name = named_dim[0]
        end = named_dim[1]
        if name in data_load_dims:
            for data_load_line in data_tensor:
                (data_load_dim, data_name, data_size, load) = data_load_line
                if data_load_dim == name:
                    action = "load" if load else "store"
                    print(prefix + action + "(" + str(data_size) + ", " + data_name + ")")

        print(prefix + "for (" + name + "=0; " + name + "<" + str(end) + "; " + name + "++)")
        prefix = prefix + prefix_inc
