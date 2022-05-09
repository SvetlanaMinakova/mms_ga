from DSE.scheduling.dnn_scheduling import DNNScheduling
from simulation.traces import SimJob


class MMSDNNInfModelSchedule:
    def __init__(self, dnn_names: [str]):
        self.dnn_names = dnn_names
        self.schedule_per_dnn_partition = {}

    def append_dnn_partition_schedule(self, dnn_name: str,
                                      partition_name: str,
                                      partition_schedule: [int]):
        """
        Append schedule of dnn partition into MMSDNNInfModelSchedule
        :param dnn_name: name of the DNN
        :param partition_name: name of the partition
        :param partition_schedule: array of integers, representing
            execution order of layers within a DNN partition.
            Layers within one partition are executed one-by-one.
            At every i-th execution step, layer with id= i-th element of
            partition_schedule is executed
        """

        if dnn_name not in self.schedule_per_dnn_partition.keys():
            self.schedule_per_dnn_partition[dnn_name] = {}
        if partition_name in self.schedule_per_dnn_partition[dnn_name].keys():
            raise Exception("MMSDNNInfModelSchedule ERROR: cannot append dnn partition schedule: it already exists!")
        else:
            self.schedule_per_dnn_partition[dnn_name][partition_name] = partition_schedule

    def print_details(self):
        for (dnn_name, schedule_per_dnn_partition) in self.schedule_per_dnn_partition.items():
            print("DNN:", dnn_name, "(", len(schedule_per_dnn_partition.items()),
                  "partitions executed in pipelined fashion )")
            print("  ", "Layers' execution order per partition:")
            for (partition_name, partition_schedule) in schedule_per_dnn_partition.items():
                print("    ", partition_name, ":", partition_schedule)


def csdf_sim_trace_schedule_to_dnn_schedule(sim_trace_schedule: [SimJob]):
    """
    Convert CSDF simulation trace schedule into execution order of CSDF actors/DNN layers
    :param sim_trace_schedule: simulation trace schedule (list of Simulation jobs)
    :return: execution order of CSDF actors/DNN layers
    """
    layers_exec_order = []
    for job in sim_trace_schedule:
        csdf_actor_name = job.task
        csdf_actor_id = int(csdf_actor_name.replace("a", ""))
        dnn_layer_id = csdf_actor_id
        layers_exec_order.append(dnn_layer_id)
    return layers_exec_order


def copy_dnn_schedule(dnn_name, src_schedule: MMSDNNInfModelSchedule, dst_schedule: MMSDNNInfModelSchedule):
    """
    Copy dnn per-partition schedule from source schedule object into destination schedule object
    :param dnn_name: name of the dnn, which per-partition schedule should be copied
    :param src_schedule:  source schedule (object of MMSDNNInfModelSchedule class)
    :param dst_schedule: destination schedule (object of MMSDNNInfModelSchedule class)
    """
    if dnn_name not in src_schedule.schedule_per_dnn_partition.keys():
        raise Exception("Cannot copy schedule: source schedule does not have specified DNN")

    dnn_schedule_per_partition = src_schedule.schedule_per_dnn_partition[dnn_name]
    for (partition_name, partition_schedule) in dnn_schedule_per_partition.items():
        dst_schedule.append_dnn_partition_schedule(dnn_name,
                                                   partition_name,
                                                   partition_schedule)

