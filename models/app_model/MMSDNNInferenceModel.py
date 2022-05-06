from DSE.low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from models.dnn_model.dnn import DNN


class MMSDNNInferenceModel:
    def __init__(self, name="app"):
        self.name = name
        self.dnn_names = [str]
        self.partitions_per_dnn = [[DNN]]
        self.schedule = None
        self.data_buffers = []
        self.phases_per_dnn_per_layer = []
        # merged with schedule?
        # self.pipeline_parallelism = None

    def __str__(self):
        return "{name: " + self.name + \
               ", dnns: " + str([dnn_name for dnn_name in self.dnn_names]) + \
               ", partitions_per_dnn" + str([len(partitions) for partitions in self.partitions_per_dnn]) +\
               "}"

    def print_details(self, print_phases=True, print_dnn_partitions=True):
        print(self.name)
        print("DNNs:")
        for dnn_id in range(len(self.dnn_names)):
            dnn_name = self.dnn_names[dnn_id]
            print(dnn_name)
            if print_phases:
                print("Phases per layer:", self.get_phases_per_layer(dnn_id))
            if print_dnn_partitions:
                partition_id = 0
                partitions = self.partitions_per_dnn[dnn_id]
                for partition in partitions:
                    print("Partition", partition_id, "of", dnn_name)
                    partition.print_details()
                    partition_id += 1

    def get_phases_per_layer(self, dnn_id: int):
        start_phase_id = 0
        end_phase_id = 0

        for prev_dnn_id in range(dnn_id):
            prev_dnn_partitions = self.partitions_per_dnn[prev_dnn_id]
            for partition in prev_dnn_partitions:
                start_phase_id += len(partition.get_layers())

        end_phase_id = start_phase_id
        dnn_partitions = self.partitions_per_dnn[dnn_id]
        for partition in dnn_partitions:
            end_phase_id += len(partition.get_layers())

        return self.phases_per_dnn_per_layer[start_phase_id: end_phase_id]







