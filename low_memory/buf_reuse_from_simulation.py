from simulation.traces import SimTrace
from models.data_buffers import DataBuffer, CSDFGDataBuffer

#######################################
# reuse buffers among multiple CSDF graphs
# (NO pipeline)
# TODO: make it pipeline-aware
def reuse_buffers_among_csdf(csdf_buffers_per_csdf):
    """
    Reuse buffers among multiple CSDF models
    :param csdf_buffers_per_csdf: csdf buffers per CSDF model
    :return: buffers reused among all the CSDF models
    """
    # Will be needed for pipeline?
    # :param sim_traces: array of simulation traces where
    #     sim_traces[i] is a trace of csdf_buffers[i], len(sim_traces)=len(csdf_buffers)

    def is_reusable(shared_buf, buf):
        """ Check if shared buffer is reusable for storage of a (new) CSDF buffer
        shared buffer is reusable for storage of a (new) CSDF buffer if
        it does not store other CSDF buffer of the same CSDF (prevent extra reuse within CSDF)
        all reuse within one CSDF should be regulated before in a separate algorithm
        """
        for shared_buf_channel in shared_buf.channels:
            for buf_channel in buf.channels:
                if channel_to_csdf_model_id[shared_buf_channel] == channel_to_csdf_model_id[buf_channel]:
                    return False
        return True

    def find_reusable_shared_buffers(buf):
        reuse_buffers = []
        for shared_buf in shared_csdf_buffers:
            if is_reusable(shared_buf, buf):
                reuse_buffers.append(shared_buf)
        return reuse_buffers

    def get_channel_to_csdf_model_id():
        ch_to_csdf_model_id = {}
        model_id = 0
        for csdf_buffers in csdf_buffers_per_csdf:
            for buf in csdf_buffers:
                for channel in buf.channels:
                    ch_to_csdf_model_id[channel] = model_id
            model_id += 1
        return ch_to_csdf_model_id

    def find_best_reusable_shared_buffer(buf, reuse_shared_buffers):
        min_cost = buf.size
        best_buffer = None
        for shared_buf in reuse_shared_buffers:
            cost = max(buf.size - shared_buf.size, 0)
            if cost < min_cost:
                min_cost = cost
                best_buffer = shared_buf
        return best_buffer

    #############
    # main script
    shared_csdf_buffers = []
    channel_to_csdf_model_id = get_channel_to_csdf_model_id()
    for csdf_buf in csdf_buffers_per_csdf:
        for buffer in csdf_buf:
            r_shared_buf = find_reusable_shared_buffers(buffer)
            if len(r_shared_buf) == 0:
                # create new shared buffer, copied from current buffer
                name = "B" + str(len(shared_csdf_buffers))
                shared_csdf_buffer = CSDFGDataBuffer(name, buffer.size)
                shared_csdf_buffers.append(shared_csdf_buffer)
            else:
                shared_csdf_buffer = find_best_reusable_shared_buffer(buffer, r_shared_buf)
            # reuse buffer to store channels
            for ch in buffer.channels:
                shared_csdf_buffer.channels.append(ch)
            # update buffer size
            shared_csdf_buffer.size = max(shared_csdf_buffer.size, buffer.size)

    return shared_csdf_buffers

#######################################
# buffers size reduction for CSDF graph


def minimize_csdfg_buf_sizes(sim_trace: SimTrace, csdf_naive_buffers):
    """
    Set CSDFG buffer sizes to minimum sizes, found using simulation
    NOTE: this step should be done after derivation of sim.trace and csdf buffers!
    :param sim_trace: simulation trace, obtained from simulation of CSDF graph execution with naive buffers
    :param csdf_naive_buffers: CSDF naive buffers, where size of every buffer is enough to store the whole
    (intermediate) data tensor
    :return:
    """
    for buf in csdf_naive_buffers:
        buf.size = sim_trace.get_max_stored_tokens(buf.name)


######################################
# buffers reuse for CSDF graph


def build_csdfg_reuse_buffers_from_sim_trace(sim_trace: SimTrace, old_csdf_buffers, mapping=None):
    """
    Get  mapping of CSDF FIFO channels onto reuse CSDF graph buffers using simulation
        :param sim_trace: simulation trace, obtained from simulation of CSDF graph execution
        Represents mapping of CSDF FIFO channels on simuluation platform buffers.
    :param old_csdf_buffers: old (not-reused) CSDF graph buffers, that were used to create simulation
    :param mapping: mapping of CNN layers/CSDF actors onto platform processors.
    CSDF buffers cannot be reused among CNN layers mapped onto different processors
    """
    def __find_generic_reuse_buf_by_user(user_name):
        for buf in generic_reuse_buf:
            if user_name in buf.users:
                return buf

    def __find_csdfg_reuse_buf_by_name(name):
        for buf in csdfg_reuse_buffers:
            if buf.name == name:
                return buf
        return None

    generic_reuse_buf = build_reuse_buffers_from_sim_trace(sim_trace, mapping)
    csdfg_reuse_buffers = []

    for old_csdf_buf in old_csdf_buffers:
        reuse_buf = __find_generic_reuse_buf_by_user(old_csdf_buf.name)
        if reuse_buf is None:
            print("NONE reuse buf for: ", old_csdf_buf.name)
        csdfg_reuse_buf = __find_csdfg_reuse_buf_by_name(reuse_buf.name)

        if csdfg_reuse_buf is None:
            csdfg_reuse_buf = CSDFGDataBuffer(reuse_buf.name, reuse_buf.size)
            csdfg_reuse_buffers.append(csdfg_reuse_buf)

        for channel in old_csdf_buf.channels:
            csdfg_reuse_buf.channels.append(channel)

    return csdfg_reuse_buffers

##########################
# generic buffers reuse


def build_reuse_buffers_from_sim_trace(sim_trace: SimTrace, mapping=None):
    """
    Build a set of reused data buffers, using application simulation trace
    :param sim_trace: (SimTrace) simulation trace
    :param mapping: mapping of CNN layers/CSDF actors onto platform processors.
    CSDF buffers cannot be reused among CNN layers mapped onto different processors
    :return: a set of DataBuffers, reused among application tasks"""

    def find_reusable_buffers(mem_name):
        """ Find all buffers that can be reused for memory with specified name"""
        r_buffers = []
        for buf in buffers:
            if buffer_reusable_for(buf, mem_name):
                r_buffers.append(buf)
        return r_buffers

    def find_best_reusable_buffer(r_buffers, new_mem_size):
        """
        Find the best reusable buffer among buffers available for reuse
        :param r_buffers: buffers available for reuse
        :param new_mem_size: size of new memory to store (in tokens)
        :return: the best reusable buffer among buffers available for reuse
        """
        if len(r_buffers) < 1:
            raise Exception("CSDF Buf reuse from simulation ERROR: I cannot choose te best buffer from an empty list!")

        best_buffer = r_buffers[0]
        min_cost = max(new_mem_size - best_buffer.size, 0)
        for buf in r_buffers:
            cost = max(new_mem_size-buf.size, 0)
            if cost < min_cost:
                min_cost = cost
                best_buffer = buf
        return best_buffer

    def buffer_reusable_for(buf, mem_name):
        # buffer can be reused for memory, unless it is already used for
        # another memory that
        # 1) is mapped on a different processor
        # 2) has overlapping occupancy intervals with new memory

        if is_storing_other_mem_mapped_on_a_different_proc(buf, mem_name):
            return False

        if is_storing_other_mem_with_overlapping_occupancy_interval(buf, mem_name):
            return False

        return True

    def get_src_and_dst_actor_id_from_mem_name(mem_name):
        """ TODO: refactoring!"""
        actor_ids_str = mem_name.replace("a", "").split("_")
        actor_ids_int = [int(layer_id) for layer_id in actor_ids_str]
        return actor_ids_int[0], actor_ids_int[1]

    def is_storing_other_mem_mapped_on_a_different_proc(buf, mem_name):
        if mapping is None:
            return False

        mem_src_actor, mem_dst_actor = get_src_and_dst_actor_id_from_mem_name(mem_name)

        for stored_memory_name in buf.users:
            stored_mem_src_actor, stored_mem_dst_actor = get_src_and_dst_actor_id_from_mem_name(stored_memory_name)
            if not stored_on_same_proc([mem_src_actor, mem_dst_actor, stored_mem_src_actor, stored_mem_dst_actor]):
                return True

        return False

    def stored_on_same_proc(actor_ids: []):
        """ Check if actors in the list are stored on the same processor.
        :return True, if actors in the list are stored on the same processor and False otherwise
        """
        if len(actor_ids) < 2:
            return True

        actors_processors = [get_actor_proc(actor_id) for actor_id in actor_ids]
        first_actor_proc = actors_processors[0]
        for actors_proc in actors_processors:
            if actors_proc != first_actor_proc:
                return False
        return True

    def get_actor_proc(actor_id):
        for proc_id in range(len(mapping)):
            if actor_id in mapping[proc_id]:
                return proc_id

    def is_storing_other_mem_with_overlapping_occupancy_interval(buf, mem_name):
        mem_occupancy_intervals = occupancy_intervals_per_mem_name[mem_name]
        for stored_memory_name in buf.users:
            stored_mem_occupancy_intervals = occupancy_intervals_per_mem_name[stored_memory_name]
            for mem_occupancy_interval in mem_occupancy_intervals:
                for stored_mem_occupancy_interval in stored_mem_occupancy_intervals:
                    if occupancy_intervals_overlap(mem_occupancy_interval.start_step,
                                                   mem_occupancy_interval.end_step,
                                                   stored_mem_occupancy_interval.start_step,
                                                   stored_mem_occupancy_interval.end_step):
                        return True
        return False

    def occupancy_intervals_overlap(start1, end1, start2, end2):
        # Check if memory occupancy intervals overlap
        # interval 2 has started while interval 1 is still active
        if start1 <= start2 <= end1:
            return True
        # interval 1 started while interval 2 is still active
        if start2 <= start1 <= end2:
            return True
        return False

    #############
    # main script

    asap_schedule = sim_trace.get_asap_schedule()
    occupancy_intervals_per_mem_name = sim_trace.generate_memory_occupancy_intervals_per_memory_first_in_last_out(asap_schedule)
    memory_names = [key for key in occupancy_intervals_per_mem_name.keys()]
    buffers = []

    for memory_name in memory_names:
        mem_size = sim_trace.get_max_stored_tokens(memory_name)
        reusable_buffers = find_reusable_buffers(memory_name)
        # no reusable buffers found
        if len(reusable_buffers) == 0:
            # create new buffer with size = memory size
            buf_name = "B" + str(len(buffers))
            reuse_buf = DataBuffer(buf_name, mem_size)
            buffers.append(reuse_buf)
        # reuse existing buffer
        else:
            reuse_buf = find_best_reusable_buffer(reusable_buffers, mem_size)
            reuse_buf.size = max(reuse_buf.size, mem_size)

        reuse_buf.assign(memory_name)

    return buffers

"""
def build_reuse_buffers_from_sim_trace(sim_trace: SimTrace):
    Build a set of reused data buffers, using application simulation trace
    :param sim_trace: (SimTrace) simulation trace
    :return: a set of DataBuffers, reused among application tasks

    def find_reusable_buffer(mem):
        for buf in buffers:
            if buffer_reusable_for(buf, mem):
                return buf

    def times_overlap(start1, end1, start2, end2):
        # Check if times overlap
        # task 2 started while task1 was executed
        if start1 <= start2 < end1:
            return True
        # task 1 started while task2 was executed
        if start2 <= start1 < end2:
            return True
        return False

    def buffer_reusable_for(buf, mem):
        # buffer can be reused for memory, unless it is already used for
        # another memory, used at the same time
        memory_times = mem_access_times[mem]

        for stored_memory in buf.users:
            stored_memory_times = mem_access_times[stored_memory]
            for (start, end) in memory_times:
                for (stored_start, stored_end) in stored_memory_times:
                    if times_overlap(start, end, stored_start, stored_end):
                        return False
        return True

    #############
    # main script
    memory_names = sim_trace.get_memory_names()
    mem_access_times = get_memory_access_times(sim_trace)
    buffers = []

    for memory_name in memory_names:
        mem_size = sim_trace.get_max_stored_tokens(memory_name)
        reuse_buf = find_reusable_buffer(memory_name)
        # create new buffer with size = memory size
        if reuse_buf is None:
            buf_name = "B" + str(len(buffers))
            reuse_buf = DataBuffer(buf_name, mem_size)
            buffers.append(reuse_buf)
        # reuse existing buffer
        else:
            reuse_buf.size = max(reuse_buf.size, mem_size)

        reuse_buf.assign(memory_name)

    return buffers
"""

def get_memory_access_times(trace: SimTrace):
    """
    Get all times, at which memory was accessed
    :param trace: simulation trace
    :return: dictionary, where key = memory_name, value = list of access times
    and every access time = (start_time, end_time), where ...
    """
    memories = trace.get_memory_names()
    tasks = trace.get_tasks()

    mem_access_times = {memory_name: [] for memory_name in memories}

    for task in tasks:
        tasks_memories = trace.get_task_memories(task)
        for memory_name in tasks_memories:
            times = trace.get_task_memory_use_time(task, memory_name)
            mem_access_times[memory_name].append(times)

    return mem_access_times




