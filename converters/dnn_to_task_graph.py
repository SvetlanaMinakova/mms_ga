from models.TaskGraph import TaskGraph, task_name_to_layer_ids, get_task_id
from models.dnn_model.dnn import set_built_in
"""
This module is building a simpleDNN graph description, used for scheduling scheduling
from dnn description
"""

"""
Get simple CNN graph example

    1 - 3
  /       \
0          5 - 6
  \       /
    2 - 4

def getExampleGraph():
    tasks = ["l_0", "l_1", "l_2", "l_3", "l_4", "l_5", "l_6"]
    tasks_num = 7
    tasks_adjacent_list = [[1, 2], [3], [4], [5], [5], [6], []]
    tasks_reverse_adjacent_list = [[], [0], [0], [1], [2], [3, 4], [5]]
    tasks_out_comm_cost = [0 for task in tasks]

    app_graph = AppGraph(layers, tasks_adjacent_list, tasks_reverse_adjacent_list, tasks_out_comm_cost)
    return app_graph
"""


def dnn_to_task_graph(dnn):
    """
    Transform a dnn into a task graph
    NOTE: DNN layers and edges in the dnn should be sorted in traverse order!
    :return:
    """

    tasks = []
    tasks_adjacent_list = []
    tasks_reverse_adjacent_list = []

    dnn_layers = dnn.get_layers()

    for layer_id in range(len(dnn_layers)):
        layer = dnn_layers[layer_id]
        layer_name = "l_" + str(layer_id)
        tasks.append(layer_name)

        l_inp_nodes_ids = []
        l_outp_nodes_ids = []

        layer_inp_connections = dnn.get_layer_input_connections(layer)
        for connection in layer_inp_connections:
            con_src = connection.src
            con_src_id = dnn.get_layer_id(con_src)
            l_inp_nodes_ids.append(con_src_id)

        layer_outp_connections = dnn.get_layer_output_connections(layer)
        for connection in layer_outp_connections:
            con_dst = connection.dst
            con_dst_id = dnn.get_layer_id(con_dst)
            l_outp_nodes_ids.append(con_dst_id)

        tasks_adjacent_list.append(l_outp_nodes_ids)
        tasks_reverse_adjacent_list.append(l_inp_nodes_ids)

    tasks_out_comm_cost = []
    for layer in dnn.get_layers():
        layer_out_tokens = layer.ofm * layer.oh * layer.ow
        tasks_out_comm_cost.append(layer_out_tokens)

    app_graph = TaskGraph(tasks, tasks_adjacent_list, tasks_reverse_adjacent_list, tasks_out_comm_cost)
    app_graph.name = dnn.name

    # add layer names as jobs
    app_graph.jobs_per_task = [[] for _ in range(len(tasks))]
    for layer_id in range(len(dnn_layers)):
        layer = dnn_layers[layer_id]
        app_graph.jobs_per_task[layer_id].append(layer.name)

    return app_graph


def dnn_to_task_graph_with_built_in(dnn, built_in_ops):
    """
    Transform a dnn into a task graph
    NOTE: DNN layers and edges in the dnn should be sorted in traverse order!
    :return:
    """
    def _connect_with_single_built_in(connections):
        if len(connections) == 1:
            if connections[0].dst.built_in:
                return True
        return False

    def _create_tasks(root_layer):
        """ Create list of tasks"""
        cur_layer = root_layer
        if cur_layer.visited:
            return

        task_name = "l_" + str(root_layer.id)
        layer_output_connections = dnn.get_layer_output_connections(cur_layer)
        while _connect_with_single_built_in(layer_output_connections):
            cur_layer.visited = True
            built_in_layer = layer_output_connections[0].dst
            task_name += "_" + str(built_in_layer.id)
            cur_layer = built_in_layer
            layer_output_connections = dnn.get_layer_output_connections(cur_layer)

        tasks.append(task_name)
        cur_layer.visited = True
        for output_connection in layer_output_connections:
            _create_tasks(output_connection.dst)

    def first_layer_id(task_name):
        layer_ids = task_name_to_layer_ids(task_name)
        return layer_ids[0]

    def last_layer_id(task_name):
        layer_ids = task_name_to_layer_ids(task_name)
        return layer_ids[0]

    def find_task_id(layer_id):
        t_id = 0
        for task_name in tasks:
            task_layer_ids = task_name_to_layer_ids(task_name)
            if layer_id in task_layer_ids:
                return t_id
            t_id += 1

    def _connect_tasks():
        """ Connect tasks, i.e., fill in tasks adjacent and reverse adjacent lists"""
        for connection in dnn.get_connections():
            src_task_id = find_task_id(connection.src.id)
            dst_task_id = find_task_id(connection.dst.id)
            if src_task_id is not None and dst_task_id is not None:
                if src_task_id != dst_task_id:
                    tasks_adjacent_list[src_task_id].append(dst_task_id)
                    tasks_reverse_adjacent_list[dst_task_id].append(src_task_id)
            else:
                raise Exception("Tasks connection error")

    def _compute_communication_costs():
        layers = dnn.get_layers()
        for task_name in tasks:
            task_output_layer_id = last_layer_id(task_name)
            task_output_layer = layers[task_output_layer_id]
            layer_out_tokens = task_output_layer.ofm * task_output_layer.oh * task_output_layer.ow
            tasks_out_comm_cost.append(layer_out_tokens)

    # main script
    set_built_in(dnn, built_in_ops)
    for layer in dnn.get_layers():
        layer.visited = False

    # create tasks
    tasks = []
    input_layer = dnn.get_input_layer()
    _create_tasks(input_layer)

    # for multi-input_examples DNN
    for layer in dnn.get_layers():
        if not layer.visited:
            _create_tasks(layer)

    tasks.sort(key=lambda task_name: first_layer_id(task_name))

    # create connections (create tasks adjacent and reverse-adjacent lists)
    tasks_adjacent_list = [[] for _ in tasks]
    tasks_reverse_adjacent_list = [[] for _ in tasks]
    _connect_tasks()

    # compute communication costs
    tasks_out_comm_cost = []
    _compute_communication_costs()

    app_graph = TaskGraph(tasks, tasks_adjacent_list, tasks_reverse_adjacent_list, tasks_out_comm_cost)
    app_graph.name = dnn.name

    # add layer names as jobs
    app_graph.jobs_per_task = [[] for _ in range(len(tasks))]
    dnn_layers = dnn.get_layers()
    for layer_id in range(len(dnn_layers)):
        layer = dnn_layers[layer_id]
        task_id = get_task_id(app_graph, layer_id)
        app_graph.jobs_per_task[task_id].append(layer.name)

    return app_graph

