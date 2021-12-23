import json


class TaskGraph:
    """
    Application graph
    Attributes:
        tasks: list of tasks (layers) names in traverse order i.e. layers =
        ["conv1_relu", "conv2_relu", "conv3_relu", "conv4_relu", "conv5_relu", "FC1_relu", "FC2_relu", "FC3", "Softmax"]
        tasks_adjacent_list - graph connectivity - outputs list
        E.g. list [[1, 2], [3], [4], [5], [5], [6], []] represents DNN graph

           1 - 3
         /       \
        0          5 - 6
         \       /
           2 - 4

        tasks_reverse_adjacent_list adjacent_list of layers input_examples connections, e.g.
        for graph above tasks_reverse_adjacent_list = [[], [0], [0], [1], [2], [3, 4], [5]]
        tasks_out_comm_cost - number of tokens to write per node
        always ends with 0 because output layer never writes
    """
    def __init__(self, tasks, tasks_adjacent_list, tasks_reverse_adjacent_list, tasks_out_comm_cost):
        self.name = "taskGraph"
        self.tasks = tasks
        self.tasks_num = len(tasks)
        self.tasks_adjacent_list = tasks_adjacent_list
        self.tasks_reverse_adjacent_list = tasks_reverse_adjacent_list
        # number of data tokens, produced by a layer/task
        self.tasks_out_comm_cost = tasks_out_comm_cost
        self.jobs_per_task = []

    def __str__(self):
        return "{name: " + self.name + ", tasks_num: " + str(self.tasks_num) +"}"

    def print_details(self, print_tasks=True, print_outputs=True, print_inputs=False, print_com_costs=False, print_jobs=True):
        print(self)
        if print_tasks:
            print("tasks:", self.tasks_num)
            for task_id in range(len(self.tasks)):
                task = self.tasks[task_id]
                print(" ", task)
                if print_jobs:
                    print("    jobs:")
                    if len(self.jobs_per_task) > task_id:
                        jobs = self.jobs_per_task[task_id]
                        for job in jobs:
                            print("     ", job)
            print("")
        if print_outputs:
            print("tasks_outputs:")
            for task_id in range(self.tasks_num):
                task_outputs = self.tasks_adjacent_list[task_id]
                print(" task", task_id, ":", task_outputs)
            print("")
        if print_inputs:
            print("tasks_inputs:")
            for task_id in range(self.tasks_num):
                task_inputs = self.tasks_reverse_adjacent_list[task_id]
                print(" task", task_id, ":", task_inputs)
        if print_com_costs:
            print("tasks_output_com_costs:")
            for task_id in range(self.tasks_num):
                com_cost = self.tasks_out_comm_cost[task_id]
                print(" task", task_id, ":", com_cost)
            print("")


def task_name_to_layer_ids(task_name):
    """
    Convert task name into
    task name should be represented using a special encoding: "l" + ("_" + layer_id)*
    where layer_id: id of DNN layer, executed within a task, and every task has one or more of "_" + layer_id
    :param task_name: encoded task name
    """
    layer_ids_str = task_name.replace("l_", "").split("_")
    layer_ids_int = [int(layer_id) for layer_id in layer_ids_str]
    return layer_ids_int


def get_task_id(app_graph, layer_id):
    """
    Get id of the task, using id of the layer
    :param app_graph: application task graph
    :param layer_id: id of the layer
    :return: id of the task
    """
    for task_id in range(app_graph.tasks_num):
        task = app_graph.tasks[task_id]
        layer_ids = task_name_to_layer_ids(task)
        if layer_id in layer_ids:
            return task_id
    raise Exception("Task not found")


def get_example_graph():
    """
    Get simple CNN graph example

        1 - 3
      /       \
    0          5 - 6
      \       /
        2 - 4
    """
    tasks = ["l_0", "l_1", "l_2", "l_3", "l_4", "l_5", "l_6"]
    tasks_adjacent_list = [[1, 2], [3], [4], [5], [5], [6], []]
    tasks_reverse_adjacent_list = [[], [0], [0], [1], [2], [3, 4], [5]]
    connections_num = 7
    tasks_out_comm_cost = [0 for _ in range(0, connections_num)]

    app_graph = TaskGraph(tasks, tasks_adjacent_list, tasks_reverse_adjacent_list, tasks_out_comm_cost)
    return app_graph

