from models.dnn_model.dnn import DNN, Layer
from converters.dnn_to_task_graph import dnn_to_task_graph
from dnn_partitioning.after_mapping.partition_dnn_with_mapping import partition_dnn_with_task_graph_and_mapping
from DSE.low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from DSE.low_memory.mms.buf_building import get_mms_buffers_multi_pipelined
from DSE.low_memory.mms.ga_based.multi_thread.MMSgaParallelMultiPipeline import MMSgaParallelMultiPipeline
from DSE.low_memory.mms.phases_derivation import get_max_phases_per_layer_per_partition_per_dnn, \
    get_phases_per_layer_per_partition_per_dnn
from models.dnn_model.transformation.external_ios_processor import external_ios_to_data_layers
from converters.json_converters.mms_chromosomes_to_json import mms_chromosomes_to_json
from util import get_project_root


def get_test_dnn_max_mem_save1():
    # create DNN
    dnn = DNN("CNN1")

    # create layers
    inp_data = Layer(32, "data", 1, 1, 3, "valid")
    inp_data.oh = inp_data.ow = 32
    inp_data.name = "l1_1(input)"
    inp_data.subop = "input"

    conv1 = Layer(32, "conv", 5, 3, 8, "same")
    conv1.oh = conv1.ow = 32
    conv1.name = "l2_1(conv)"

    conv2 = Layer(32, "conv", 5, 8, 8, "same")
    conv2.oh = conv2.ow = 32
    conv2.name = "l3_1(conv)"

    add = Layer(32, "arithmetic", 1, 16, 16, "valid")
    add.oh = conv1.ow = 32
    add.subop = "add"
    add.name = "l4_1(add)"

    outp_data = Layer(32, "relu", 1, 16, 16, "valid")
    outp_data.oh = outp_data.ow = 32
    outp_data.name = "l5_1(output)"
    outp_data.subop = "output"

    # add layers and connections
    dnn.stack_layer(inp_data)
    dnn.stack_layer(conv1)
    dnn.stack_layer(conv2)
    dnn.stack_layer(add)
    dnn.stack_layer(outp_data)

    # add residual connection
    dnn.connect_layers_by_name("l2_1(conv)", "l4_1(add)")
    # dnn.set_auto_ios()

    return dnn


def get_test_dnn_max_mem_save2():
    # create DNN
    dnn = DNN("CNN2")

    # create layers
    inp_data = Layer(32, "data", 1, 1, 3, "valid")
    inp_data.oh = inp_data.ow = 32
    inp_data.name = "l1_2(input)"
    inp_data.subop = "input"

    conv1 = Layer(32, "conv", 5, 3, 8, "valid")
    conv1.oh = conv1.ow = 28
    conv1.name = "l2_2(conv)"

    fc = Layer(28, "gemm", 1, 8, 10, "valid")
    fc.oh = fc.ow = 1
    fc.subop = "gemm"
    fc.name = "l3_2(gemm)"

    outp_data = Layer(1, "data", 1, 10, 10, "valid")
    outp_data.oh = outp_data.ow = 1
    outp_data.name = "l4_2(output)"
    outp_data.subop = "output"

    # add layers and connections
    dnn.stack_layer(inp_data)
    dnn.stack_layer(conv1)
    dnn.stack_layer(fc)
    dnn.stack_layer(outp_data)
    # dnn.set_auto_ios()

    return dnn


def example_app_with_manual_chromosome():
    """ Multi-CNN example where CNN 1 and CNN 2 are executed, and CNN 2 is executed with pipeline parallelism"""
    cnn1 = get_test_dnn_max_mem_save1()
    cnn2 = get_test_dnn_max_mem_save2()

    # handcrafted mapping where cnn is executed in a pipelined manner
    # as two partitions: (layer0, layer1) partition and (layer2, layer2) partition
    handcrafted_mapping = [[0, 1], [2, 3]]

    # task graph
    cnn2_task_graph = dnn_to_task_graph(cnn2)

    # dnn_partitioning
    cnn2_partitions, cnn2_connections = partition_dnn_with_task_graph_and_mapping(cnn2,
                                                                                  cnn2_task_graph,
                                                                                  handcrafted_mapping)
    layers_num = 0

    # total application partitions per dnn
    partitions_per_dnn = [[cnn1], cnn2_partitions]
    for partitions in partitions_per_dnn:
        for partition in partitions:
            external_ios_to_data_layers(partition)
            layers_num += len(partition.get_layers())

    # create chromosome
    chromosome = MMSChromosome(layers_num)
    # chromosome.dp_by_parts = [True, True, False, True, True, True, True, False, False]
    # chromosome.dp_by_parts = [True, True, True, True, False, False, False, True, True, False, False]
    # chromosome.dp_by_parts = [False, False, True, True, True, False, True, False, True, True, True]

    chromosome.dp_by_parts = [False, False, True, True, True, False, False, False, False, False, False]

    # without external I/Os
    # chromosome.dp_by_parts = [False, False, True, True, True, False, False, False, False]

    # max phases
    max_phases = get_max_phases_per_layer_per_partition_per_dnn(partitions_per_dnn)
    phases = get_phases_per_layer_per_partition_per_dnn(partitions_per_dnn, chromosome.dp_by_parts, max_phases)

    print("Phases")

    buffers = get_mms_buffers_multi_pipelined(partitions_per_dnn, phases)
    for buffer in buffers:
        buffer.print_details()
        print()

    sum_buf_size = sum([buf.size for buf in buffers])
    print("Total buffers size: ", sum_buf_size)


def example_app_with_ga():
    """ Multi-CNN example where CNN 1 and CNN 2 are executed, and CNN 2 is executed with pipeline parallelism"""
    cnn1 = get_test_dnn_max_mem_save1()
    cnn2 = get_test_dnn_max_mem_save2()

    # handcrafted mapping where cnn is executed in a pipelined manner
    # as two partitions: (layer0, layer1) partition and (layer2, layer2) partition
    handcrafted_mapping = [[0, 1], [2, 3]]

    # task graph
    cnn2_task_graph = dnn_to_task_graph(cnn2)

    # dnn_partitioning
    cnn2_partitions, cnn2_connections = partition_dnn_with_task_graph_and_mapping(cnn2,
                                                                                  cnn2_task_graph,
                                                                                  handcrafted_mapping)
    layers_num = 0

    # total application partitions per dnn
    partitions_per_dnn = [[cnn1], cnn2_partitions]
    for partitions in partitions_per_dnn:
        for partition in partitions:
            external_ios_to_data_layers(partition)

    ga = MMSgaParallelMultiPipeline(partitions_per_dnn, 100, 500, 66, 0.5, 5, 20, 0.5, 1*10e6, 5)
    ga.init_with_random_population()
    pareto = ga.run()

    output_file_path = str(get_project_root()) + "/output/example_app.json"
    mms_chromosomes_to_json(pareto, output_file_path)

example_app_with_manual_chromosome()
# example_app_with_ga()