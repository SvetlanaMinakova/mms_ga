from DSE.low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from models.dnn_model.dnn import DNN
from DSE.scheduling.mms_dnn_inf_model_schedule import MMSDNNInfModelSchedule
from models.data_buffers import DataBuffer


class MMSDNNInferenceModel:
    def __init__(self,
                 app_name: str,
                 dnn_names: [str],
                 partitions_per_dnn: [[DNN]],
                 schedule: MMSDNNInfModelSchedule,
                 data_buffers: [DataBuffer],
                 phases: []):
        self.name = app_name
        self.dnn_names = dnn_names
        self.partitions_per_dnn = partitions_per_dnn
        self.schedule = schedule
        self.data_buffers = data_buffers
        self.phases_per_dnn_per_layer = phases

    def __str__(self):
        return "MMSDNNInferenceModel: {name: " + self.name + \
               ", dnns: " + str([dnn_name for dnn_name in self.dnn_names]) + \
               ", partitions_per_dnn" + str([len(partitions) for partitions in self.partitions_per_dnn]) +\
               "}"

    def print_details(self, print_phases=True,
                      print_dnn_partitions=False,
                      print_schedule=True,
                      print_buffers=True):
        print(self)

        if print_dnn_partitions:
            print("Partitions per dnn: ")
            for dnn_id in range(len(self.dnn_names)):
                dnn_name = self.dnn_names[dnn_id]
                print(dnn_name)
                partition_id = 0
                partitions = self.partitions_per_dnn[dnn_id]
                for partition in partitions:
                    print("Partition", partition.name, "(", (partition_id+1), "/", len(partitions), ") of", dnn_name)
                    partition.print_details()
                    partition_id += 1
            print()

        if print_phases:
            print("Phases (per dnn):")
            for dnn_id in range(len(self.dnn_names)):
                dnn_name = self.dnn_names[dnn_id]
                print("  ", dnn_name, ":", self.get_phases_per_layer(dnn_id))
            print()

        if print_schedule:
            print("Schedule:")
            self.schedule.print_details()
            print()

        if print_buffers:
            print("Buffers:")
            for buffer in self.data_buffers:
                buffer.print_details()

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






