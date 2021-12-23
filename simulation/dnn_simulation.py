from models.dnn_model.dnn import DNN
from models.edge_platform.SimulationPlatform import SimulationPlatform, SimulationBuffer, SimulationProc
from simulation.traces import SimTrace, SimJob, SimMemoryAccess


def simulate_execution_sequential(dnn: DNN,
                                  dnn_buffers,
                                  trace_memory_access=True,
                                  verbose=True):
    """
    Simulate sequential (layer-by-layer) execution of a DNN
    :param dnn: DNN to execute
    :param dnn_buffers: list of DNN buffers, used to store
    data, exchanged between the DNN layers.
    :param trace_memory_access (flag) If True, memory access will be added to the trace
    :param verbose: print details
    :return: trace (SimTrace), which describes simulation in time
    """

    def fire_layer(layer_id, processor):
        """
        Fire (execute) DNN layer
        when layer fires, it :
          1) consumes input_examples data from all input_examples edges (buffers)
          2) performs an operator
          3) writes output data to all its output edges (buffers)
        :param layer_id: id of the layer
        :param processor: processor, where layer is executed
        """
        # prepare data
        layer = layers[layer_id]
        task_desc = layer.name
        job_desc = layer.subop
        # estimate time
        start_time = trace.get_proc_time(processor.name)
        layer_exec_time = layer.time_eval
        end_time = start_time + layer_exec_time

        ###########################
        # simulate task execution

        if verbose:
            print("EXECUTE layer", layer_id, " start time: ", start_time)

        # read input_examples data
        for input_edge in dnn.get_layer_input_connections(layer):
            input_buffer = channel_to_buf_mapping[input_edge]
            data_src = input_edge.src
            cons_rate = data_src.ofm * data_src.oh * data_src.ow
            input_buffer.read_tokens(cons_rate)
            if trace_memory_access:
                mem_access = SimMemoryAccess(task_desc,
                                             job_desc,
                                             input_buffer.name,
                                             "read",
                                             cons_rate,
                                             start_time,
                                             end_time)
                trace.add_mem_access(mem_access)

        # execute task
        job = SimJob(task_desc, job_desc, processor.name, start_time, end_time)
        trace.add_job(job)
        # free processor, where task was executed
        processor.free()

        # write output data to output channels
        for output_edge in dnn.get_layer_output_connections(layer):
            output_buffer = channel_to_buf_mapping[output_edge]
            data_src = layer
            prod_rate = layer.ofm * layer.oh * layer.ow
            output_buffer.write_tokens(prod_rate)
            if trace_memory_access:
                mem_access = SimMemoryAccess(task_desc, job_desc, output_buffer.name, "write", prod_rate, start_time,
                                             end_time)
                trace.add_mem_access(mem_access)

        # increase actor's phase
        if verbose:
            print("layer", layer.name, " is fired ")
            print(" end time: ", end_time)

    # main script
    ###################
    # prepare variables
    # CSDFG
    layers = dnn.get_layers()
    edges = dnn.get_connections()
    layers_num = len(layers)
    # platform
    platform = create_simulation_platform(dnn_buffers)
    sim_buffers = platform.get_buffers()
    processors = platform.get_processors()
    # mapping of DNN channels onto simulation platform buffers
    channel_to_buf_mapping = get_edge_to_sim_buf_mapping(dnn, edges, sim_buffers, dnn_buffers)
    # trace
    trace = SimTrace()

    # all on one processor
    proc = processors[0]

    # simulate layer exec
    for l_id in range(layers_num):
        fire_layer(l_id, proc)


def create_simulation_platform(dnn_buffers, proc_num=1):
    """
    :param dnn_buffers: list of DNN buffers, used to store
    data, exchanged between the DNN layers.
    :param proc_num: number of processors in the target simulaton platform
    :return: simulation platform
    """
    platform = SimulationPlatform("SimPla")
    for proc_id in range(proc_num):
        proc_name = "proc" + str(proc_id)
        proc = SimulationProc(proc_name)
        platform.add_processor(proc)

    for dnn_buf in dnn_buffers:
        sim_buf = dnn_buf_to_platform_sim_buf(dnn_buf)
        platform.add_buffer(sim_buf)

    return platform


def dnn_buf_to_platform_sim_buf(dnn_buf):
    """
    Convert DNN buffer into platform simulation buffer
    :param dnn_buf: DNN buffer
    :return: platform simulation buffer
    """
    sim_buf = SimulationBuffer(dnn_buf.name, dnn_buf.size)
    return sim_buf


def get_edge_to_sim_buf_mapping(dnn, edges, sim_buffers, dnn_buffers):
    """
    Generate mapping of DNN model edges onto the simulation buffers
    :param dnn: DNN
    :param edges:  DNN model edges (connections)
    :param sim_buffers: simulation buffers
    :param dnn_buffers: list of DNN buffers, used to store
    data, exchanged between the DNN layers
    :return: dictionary, where key= DNN edge (connection), value = simulation buffer,
    allocated to store data of the DNN edge
    """
    def __get_dnn_buffer(edge):
        for dnn_buf in dnn_buffers:
            if dnn in dnn_buf.connections_per_dnn.keys():
                if edge in dnn_buf.connections_per_dnn[dnn]:
                    return dnn_buf

    def __find_sim_buf_by_name(name):
        for sim_buf in sim_buffers:
            if sim_buf.name == name:
                return sim_buf

    # main script
    edge_to_sim_buf_mapping = {}
    for e in edges:
        dnn_buffer = __get_dnn_buffer(e)
        sim_buffer = __find_sim_buf_by_name(dnn_buffer.name)
        edge_to_sim_buf_mapping[e] = sim_buffer

    return edge_to_sim_buf_mapping
