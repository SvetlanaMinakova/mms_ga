from DSE.low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from models.dnn_model.dnn import DNN


class MMSFinalAppModel:
    def __init__(self, name="app"):
        self.name = name
        self.dnns = []
        self.partitions_per_dnn = [[]]
        self.schedule = None
        self.data_buffers = []
        self.phases_per_dnn_per_layer = []
        # merged with schedule?
        # self.pipeline_parallelism = None

    def __str__(self):
        return "{name: " + self.name + \
               ", dnns: " + str([dnn.name for dnn in self.dnns]) + \
               ", partitions_per_dnn" + str([len(partitions) for partitions in self.partitions_per_dnn]) +\
               "}"

    def print_details(self):
        print(str(self))


def mms_chromosome_to_final_app_model(app_name, chromosome: MMSChromosome):
    app_model = MMSFinalAppModel(app_name)
    app_model.print_details()



