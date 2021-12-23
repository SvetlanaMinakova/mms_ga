import random
import sys


class MMSChromosome:
    """
    Chromosome that represents DP (data processing by parts) within CNN
    """
    def __init__(self, layers_num):
        self.layers_num = layers_num
        # list of boolean flags of len = len(self.layers_num)
        # each i-th flag = True/False determines whether layer li of dnn
        # processes data by parts (True) or not (False)
        self.dp_by_parts = [False for _ in range(self.layers_num)]
        # chromosome is characterised with loss of time, caused
        # by using data processing by parts as well as by
        # total size of buffers (in MegaBytes)
        # after just being initialized, chromosome has infinite
        # time loss and buffers size. The real time loss and buffers size are
        # estimated and assigned to the chromosome during GA
        self.time_loss = sys.maxsize
        self.buf_size = sys.maxsize

        # related dnn/partitions per dnn
        self.dnn = None
        self.partitions_per_dnn = None

    def init_random(self, dp_by_parts_init_probability=0.5):
        """
        Init a chromosome randomly
        :param dp_by_parts_init_probability: float number 0 <=x <= 1:
        probability with which (every) layer li in the dnn
        processes data by parts or not. By default, = 0.5 (probability of
        randomly choosing from two possibilities: processing by parts or no processing by parts)
        """
        for layer_id in range(self.layers_num):
            # roll a dice: should the layer process data by parts?
            random_chance = random.uniform(0, 1)  # Random float x, 0 <= x < 1
            if random_chance <= dp_by_parts_init_probability:
                # the layer processes data by parts
                self.dp_by_parts[layer_id] = True
            else:
                # the layer does not process data by parts
                self.dp_by_parts[layer_id] = False

    def mutate(self):
        random_layer_id = random.randint(0, self.layers_num - 1)  # get random layer
        self.dp_by_parts[random_layer_id] = not self.dp_by_parts[random_layer_id]
        # print("Mutate: layer ", random_layer_id, "processing by parts inverted")

    def clean(self):
        self.dp_by_parts = [False for _ in range(self.layers_num)]

    def __str__(self):
        max_phases = len([dp for dp in self.dp_by_parts if dp is True])
        return "{layers with phases: " + str(max_phases) + \
               ", time_loss: " + str(self.time_loss) \
               + ", buf_size " + str(self.buf_size) + "}"

    def print_short(self):
        print(self)

    def print_long(self):
        str_long = "{ dp by parts: " + str(self.dp_by_parts) + \
                   ", time loss: " + str(self.time_loss) + \
                   ", buf size " + str(self.buf_size) + "}"
        print(str_long)


