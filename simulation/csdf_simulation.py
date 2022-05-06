from models.csdf_model.csdf import CSDFGraph
from models.edge_platform.SimulationPlatform import SimulationPlatform, SimulationBuffer, SimulationProc
from simulation.traces import SimTrace, SimJob, SimMemoryAccess


def simulate_execution_asap(csdfg:CSDFGraph,
                            csdfg_buffers,
                            proc_num=1,
                            max_samples=1,
                            trace_memory_access=True,
                            verbose=True):
    """
    Simulate execution of CSDF model where every actor is executed as soon as possible
    :param csdfg: CSDFGraph for simulation
    :param csdfg_buffers: list of CSDF graph buffers, used to store
        data, exchanged though the CSDF graph channels.
    :param max_samples: maximum number of input_examples samples (e.g. input_examples images) to process
    :param proc_num: number of processors to simulate execution on
    :param trace_memory_access (flag) If True, memory access will be added to the trace
    :param verbose: print details
    :return: trace (SimTrace), which describes simulation in time
    """

    def get_next_free_processor():
        for processor in processors:
            if processor.is_free():
                return processor
        return None

    def get_next_ready_actor_id(cur_actor_id):
        # first visit all the following actors
        for actor_id in range((cur_actor_id + 1), len(actors)):
            if is_actor_ready(actor_id):
                next_ready_actor_id = actor_id
                return next_ready_actor_id

        # if actor is not found, visit all previous actors
        for actor_id in range(0, cur_actor_id+1):
            if is_actor_ready(actor_id):
                next_ready_actor_id = actor_id
                return next_ready_actor_id

        return -1

    def is_actor_ready(actor_id):
        # actor is ready to fire, when all his input_examples buffers have enough data for actor to consume

        actor = actors[actor_id]
        phase = phase_per_actor[actor_id]
        # too many phases (for any actor)
        # if actor_id == 0 and phase >= max_src_phases:
        if phase >= actors[actor_id].phases * max_samples:
            return False

        for input_channel in csdfg.get_input_channels(actor):
            input_buffer = channel_to_buf_mapping[input_channel]
            cons_rate = input_channel.cons_seq[phase % actor.phases]
            if input_buffer.occupied < cons_rate:
                return False
        return True

    def fire_actor(actor_id, processor):
        """
        Fire actor
        when actor fires, it :
          1) consumes input_examples data from all input_examples channels (buffers)
          2) performs an operator
          3) writes output data to all its output channels (buffers)
        :param actor_id: id of the actor
        :param processor: processor, where actors is executed
        """
        # prepare data
        actor = actors[actor_id]
        phase = phase_per_actor[actor_id]
        task_desc = actor.name
        # job_desc = ("phase " + str(max(phase - 1, 0) % actor.phases + 1) + "/" + str(actor.phases))
        job_desc = ("phase " + str((phase % actor.phases)+1) + "/" + str(actor.phases))

        # estimate time
        start_time = trace.get_proc_time(processor.name)
        actor_exec_time = actor.time_per_phase[phase % actor.phases]
        end_time = start_time + actor_exec_time

        ###########################
        # simulate task execution

        if verbose:
            print("EXECUTE actor", actor_id, " start time: ", start_time)

        # read input_examples data
        for input_channel in csdfg.get_input_channels(actor):
            input_buffer = channel_to_buf_mapping[input_channel]
            cons_rate = input_channel.cons_seq[phase % actor.phases]
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
        for output_channel in csdfg.get_output_channels(actor):
            output_buffer = channel_to_buf_mapping[output_channel]
            prod_rate = output_channel.prod_seq[phase % actor.phases]
            output_buffer.write_tokens(prod_rate)
            if trace_memory_access:
                mem_access = SimMemoryAccess(task_desc, job_desc, output_buffer.name, "write", prod_rate, start_time, end_time)
                trace.add_mem_access(mem_access)

        # increase actor's phase
        phase_per_actor[actor_id] = phase_per_actor[actor_id] + 1
        if verbose:
            print("actor", actor.name, " fired phase", (phase % actor.phases) + 1, "/", actor.phases)
            print(" end time: ", end_time)

    def schedule_next_actor(last_actor_id=-1):
        ready_actor_id = get_next_ready_actor_id(last_actor_id)
        processor = get_next_free_processor()
        return ready_actor_id, processor

    def check_consistency():
        # check that every actor has executed its final phase
        for i in range(len(actors)):
            phases_performed = phase_per_actor[i]
            phases_expected = actors[i].phases * max_samples
            if phases_performed != phases_expected:
                if verbose:
                    print("Sim inconsistency: actor", actors[i], "performed phases",
                          phases_performed, "/", phases_expected, "expected")
                return False
        return True

    # main script
    ###################
    # prepare variables
    # CSDFG
    actors = csdfg.get_actors()
    channels = csdfg.get_channels()
    actors_num = len(actors)
    phase_per_actor = [0 for _ in range(actors_num)]
    # platform
    platform = create_simulation_platform(csdfg_buffers, proc_num)
    sim_buffers = platform.get_buffers()
    processors = platform.get_processors()
    # mapping of CSDFG channels onto simulation platform buffers
    channel_to_buf_mapping = get_channel_to_sim_buf_mapping(channels, sim_buffers, csdfg_buffers)
    # trace
    trace = SimTrace()
    # check that every actor has executed its final phase


    ################
    # run simulation
    # restrict number of phases of the first actor, so that only max_samples are executed
    # max_src_phases = actors[0].phases * max_samples

    # schedule first actor
    last_executed_actor_id = -1
    next_actor_id, next_proc = schedule_next_actor(last_executed_actor_id)
    # print("first scheduled actor: ", next_actor_id)

    # execute current actor and schedule next actor
    while next_actor_id != -1 and next_proc is not None:
        fire_actor(next_actor_id, next_proc)
        last_executed_actor_id = next_actor_id
        next_actor_id, next_proc = schedule_next_actor(last_executed_actor_id)

    consistent = check_consistency()
    if verbose:
        print("Simulation consistent: ", consistent)

    return trace


def get_channel_to_sim_buf_mapping(channels, sim_buffers, csdfg_buffers):
    """
    Generate mapping of CSDFG model channels onto the simulation buffers
    :param channels:  CSDFG model channels
    :param sim_buffers: simulation buffers
    :param csdfg_buffers: list of CSDF graph buffers, used to store
    data, exchanged though the CSDF graph channels.
    :return: dictionary, where key=CSDF channel, value = simulation buffer,
    allocated to store data of the CSDF channel
    """
    def __get_csdfg_buffer(channel):
        for csdfg_buf in csdfg_buffers:
            if channel in csdfg_buf.channels:
                return csdfg_buf

    def __find_sim_buf_by_name(name):
        for sim_buf in sim_buffers:
            if sim_buf.name == name:
                return sim_buf

    # main script
    channel_to_sim_buf_mapping = {}
    for ch in channels:
        csdfg_buffer = __get_csdfg_buffer(ch)
        sim_buffer = __find_sim_buf_by_name(csdfg_buffer.name)
        channel_to_sim_buf_mapping[ch] = sim_buffer

    return channel_to_sim_buf_mapping


def create_simulation_platform(csdfg_buffers, proc_num=1):
    """
    :param proc_num: number of processors
    :param csdfg_buffers: list of CSDF graph buffers, used to store
    data, exchanged though the CSDF graph channels.
    :return: simulation platform
    """
    platform = SimulationPlatform("SimPla")
    for proc_id in range(proc_num):
        proc_name = "proc" + str(proc_id)
        proc = SimulationProc(proc_name)
        platform.add_processor(proc)

    for csdfg_buf in csdfg_buffers:
        sim_buf = csdfg_buf_to_platform_sim_buf(csdfg_buf)
        platform.add_buffer(sim_buf)

    return platform


def csdfg_buf_to_platform_sim_buf(csdfg_buf):
    """
    Convert CSDF buffer into platform simulation buffer
    :param csdfg_buf: CSDFG buffer
    :return: platform simulation buffer
    """
    sim_buf = SimulationBuffer(csdfg_buf.name, csdfg_buf.size)
    return sim_buf

