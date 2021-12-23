from models.app_model.DNNBasedAppModel import InterDNNConnection
from models.dnn_model.dnn import DNN
from models.TaskGraph import TaskGraph, task_name_to_layer_ids
import copy


def partition_dnn_with_task_graph_and_mapping(dnn: DNN, task_graph: TaskGraph, mapping: []):
    """
    Partition dnn according to the task graph, generated from this DNN
    :param dnn: dnn, represented as (analysis) dnn model
    :param task_graph: task graph, generated from this DNN
    :param mapping: [tasks_per_processor[i]], i in [1, len(processors_num)], where
     processors_num is the total number of processors available on the platform,
     tasks_per_processor[i] = [task_1, task_2, ..., task_t] is the list of tasks mapped on the processor i,
     where task_t is id of task in the task graph

    :return: tuple: partitions, connections where:
        partitions is a  list of dnn partitions, where every partition is a DNN, that is
        a sub-graph of the original dnn, and all partitions together represent
        functionality of the original DNN
        connections: connections between the DNN partitions
    """
    partitioner = DNNPartitioner(dnn, task_graph, mapping)
    partitioner.partition()
    partitions = partitioner.get_partitions()
    connections = partitioner.get_inter_partition_connections()
    return partitions, connections


class DNNPartitioner:
    """
    Partitions DNN into sub-graphs (partitions)
    according to the task graph, created from the DNN
    """
    def __init__(self, dnn: DNN, task_graph: TaskGraph, mapping: []):
        self.dnn = dnn
        self.layers = dnn.get_layers()
        self.task_graph = task_graph
        self.mapping = mapping

        # meta-data
        self.__partitions = []
        self.__inter_partition_connections = []

    def partition(self):
        self.__create_partitions()
        self.__add_connections_within_partitions()
        self.__add_external_ios()
        self.__transfer_external_ios()

    def __create_partitions(self):
        self.__partitions = []
        for task_ids_per_processor in self.mapping:
            task_names_per_processor = []
            for task_id in task_ids_per_processor:
                task_name = self.task_graph.tasks[task_id]
                task_names_per_processor.append(task_name)

            partition = self.__create_partition(task_names_per_processor)
            self.__partitions.append(partition)

    def __create_partition(self, task_names):
        partition = DNN(name="Subnet" + str(len(self.__partitions)))
        for task_name in task_names:
            task_layer_ids = task_name_to_layer_ids(task_name)
            for layer_id in task_layer_ids:
                layer = self.layers[layer_id]
                layer_copy = copy.deepcopy(layer)
                partition.add_layer(layer_copy)

        return partition

    def __add_connections_within_partitions(self):
        for connection in self.dnn.get_connections():
            src_partition_id = self.__find_partition_id(connection.src.id)
            dst_partition_id = self.__find_partition_id(connection.dst.id)
            # connection is within partitions
            if src_partition_id == dst_partition_id:
                partition = self.__partitions[src_partition_id]
                partition.connect_layers_by_name(connection.src.name, connection.dst.name)

    def __transfer_external_ios(self):
        """ Transfer external I/Os from original DNN to partitions"""
        self.__transfer_external_inputs()
        self.__transfer_external_outputs()

    def __transfer_external_inputs(self):
        """ Transfer external inputs (data sources) from original DNN to partitions"""
        inputs = self.dnn.get_inputs()
        for external_input in inputs:
            layer_id = external_input.dnn_layer.id
            partition_id = self.__find_partition_id(layer_id)
            partition = self.__partitions[partition_id]
            name = external_input.data_layer.name
            iw = external_input.data_layer.iw
            ih = external_input.data_layer.ih
            ifm = external_input.data_layer.ifm
            partition.add_external_input(name, iw, ih, ifm)

    def __transfer_external_outputs(self):
        """ Transfer external outputs (data consumers) from original DNN to partitions"""
        outputs = self.dnn.get_outputs()
        for external_output in outputs:
            layer_id = external_output.dnn_layer.id
            partition_id = self.__find_partition_id(layer_id)
            partition = self.__partitions[partition_id]
            name = external_output.data_layer.name
            ow = external_output.data_layer.ow
            oh = external_output.data_layer.oh
            ofm = external_output.data_layer.ofm
            partition.add_external_output(name, ow, oh, ofm)

    def __add_external_ios(self):
        """ Add external I/Os that occur due to communication between partitions"""
        for connection in self.dnn.get_connections():
            src_partition_id = self.__find_partition_id(connection.src.id)
            dst_partition_id = self.__find_partition_id(connection.dst.id)

            # connection is among partitions (not within one)
            if src_partition_id != dst_partition_id:
                io_name = "external_" + connection.src.name + "_" + connection.dst.name
                src_partition = self.__partitions[src_partition_id]
                dst_partition = self.__partitions[dst_partition_id]
                # TODO:check
                # add external output to the source layer in the source partition
                src_layer_in_partition = src_partition.find_layer_by_name(connection.src.name)
                src_partition.add_external_output(io_name,
                                                  connection.src.ow,
                                                  connection.src.oh,
                                                  connection.src.ofm,
                                                  src_layer_in_partition)
                # add external input to the destination layer in the destination partition
                dst_layer_in_partition = dst_partition.find_layer_by_name(connection.dst.name)
                dst_partition.add_external_input(io_name,
                                                 connection.dst.iw,
                                                 connection.dst.ih,
                                                 connection.dst.ifm,
                                                 dst_layer_in_partition)

                # add external I/O
                inter_partition_connection = InterDNNConnection(io_name,
                                                                src_partition, dst_partition,
                                                                connection.src.ow, connection.src.oh,
                                                                connection.src.ofm)
                self.__inter_partition_connections.append(inter_partition_connection)

    def __find_partition_id(self, layer_id):
        for proc_id in range(len(self.mapping)):
            partition_id = proc_id
            task_ids_per_processor = self.mapping[proc_id]
            task_names_per_processor = [self.task_graph.tasks[task_id] for task_id in task_ids_per_processor]
            for task_name in task_names_per_processor:
                layer_ids = task_name_to_layer_ids(task_name)
                if layer_id in layer_ids:
                    return partition_id

    #########
    # getters

    def get_partitions(self):
        return self.__partitions

    def get_inter_partition_connections(self):
        return self.__inter_partition_connections

    #######
    # print functions

    def print_partitions(self):
        for partition in self.__partitions:
            print("PARTITION: ")
            partition.print_details()
            print("")