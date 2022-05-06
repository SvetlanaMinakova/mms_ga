from DSE.scheduling.dnn_scheduling import DNNScheduling
from simulation.traces import SimJob


class MMSDNNInfModelSchedule:
    def __init__(self, dnn_names: [str],
                 inter_partition_schedule_type=DNNScheduling.SEQUENTIAL):
        self.dnn_names = dnn_names
        self.schedule_per_dnn_partition = {}
        self.inter_partition_schedule_type = inter_partition_schedule_type

    def append_dnn_partition_schedule(self, dnn_name: str,
                                      partition_name: str,
                                      partition_schedule: [int]):

        if dnn_name not in self.schedule_per_dnn_partition.keys():
            self.schedule_per_dnn_partition[dnn_name] = {}
        if partition_name in self.schedule_per_dnn_partition[dnn_name].keys():
            raise Exception("MMSDNNInfModelSchedule ERROR: cannot append dnn partition schedule: it already exists!")
        else:
            self.schedule_per_dnn_partition[dnn_name][partition_name] = partition_schedule


def csdf_sim_trace_schedule_to_layers_exec_order(sim_trace_schedule: [SimJob]):
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


