class Architecture:
    """
    Target architecture class
    :param processors: list of distinct processors names, e.g., ["CPU0", "CPU1", "CPU2", "CPU3", "CPU4", "GPU"]
    :param processors_types: list of processor types, e.g., ["large_CPU", "large_CPU", "small_CPU", "small_CPU", "GPU"]
        required: len(processors_types) = len(processors)
    :param processor_types_distinct: list of distinct processor types, e.g., ["large_CPU", "small_CPU", "GPU"]
    """
    def __init__(self, processors, processors_types,  processor_types_distinct):
        self.processors = processors
        # names of processors that are accelerators
        self.accelerators = []
        self.processors_types = processors_types
        self.processors_types_distinct = processor_types_distinct
        self.processor_types_distinct_num = len(processor_types_distinct)
        self.processors_num = len(processors)
        self.communication_channels = []
        # max flops for every type of processor: needed for flop-based evaluation
        self.max_giga_flops_per_proc_type = [1 for _ in processor_types_distinct]

    def add_communication_channel(self, proc_type_1, proc_type_2, speed):
        """
        Add communication speed (in tokens/s) between two processors
        :param proc_type_1: type of the source processor (unique)
        :param proc_type_2: type of the destination processor (unique)
        :param speed: communication speed (in tokens/s) between processors of specified types
        """
        channel = CommunicationChannel(proc_type_1, proc_type_2, speed)
        self.communication_channels.append(channel)

    def find_communication_channel(self, proc_type_1, proc_type_2):
        for channel in self.communication_channels:
            if proc_type_1 in channel.processors and proc_type_2 in channel.processors:
                return channel
        return None

    def get_communication_speed_mb_s(self, proc_type_1, proc_type_2):
        channel = self.find_communication_channel(proc_type_1, proc_type_2)
        if channel is None:
            return 0
        return channel.bandwidth_mb_s

    def get_proc_type_id(self, processor_id):
        """
        Get id of distinct processor type by processor id
        :param processor_id: processor id
        :return: id of distinct processor type
        """
        processor_type = self.processors_types[processor_id]
        for i in range(len(self.processors_types_distinct)):
            if processor_type == self.processors_types_distinct[i]:
                return i
        return None

    def get_max_giga_flops_for_proc_type(self, proc_type_id):
        """
        Get max flops for processor
        :param proc_type_id: distinct processor type id
        :return: max flops for processor
        """
        if len(self.max_giga_flops_per_proc_type) > proc_type_id:
            return self.max_giga_flops_per_proc_type[proc_type_id]
        return 1

    def get_proc_id_by_name(self, name):
        for proc_id in range(len(self.processors)):
            if self.processors[proc_id] == name:
                return proc_id

    def get_first_accelerator_proc_id(self):
        if self.accelerators:
            accelerator_name = self.accelerators[0]
            first_accelerator_id = self.get_proc_id_by_name(accelerator_name)
            return first_accelerator_id
        return -1


class CommunicationChannel:
    """
    Communication channel between two types of processors
    characterises speed with which two processors of specified types communicate
    Attributes:
        processor_type_1, processor_type_2: types of processors communicating
        bandwidth_mb_s : communication bandwidth (in MegaBytes per second)
    """
    def __init__(self, processor_type_1, processor_type_2, bandwidth_mb_s):
        self.processors = [processor_type_1, processor_type_2]
        self.bandwidth_mb_s = bandwidth_mb_s


def get_jetson():
    """
    Get Jetson as architecture example
    """
    # baseline architecture
    processors = ["CPU0", "CPU1", "CPU2", "CPU3", "CPU4", "GPU"]
    processor_types = ["large_CPU", "large_CPU", "small_CPU", "small_CPU", "small_CPU", "GPU"]
    processor_types_distinct = ["large_CPU", "small_CPU", "GPU"]
    jetson = Architecture(processors, processor_types, processor_types_distinct)

    # bandwidth between processors
    # CPU/GPU bandwidth = 20 GB/s
    cpu_gpu_bandwidth_mb_s = 20 * 1e9/1e6
    jetson.add_communication_channel("large_CPU", "GPU", cpu_gpu_bandwidth_mb_s)
    jetson.add_communication_channel("small_CPU", "GPU", cpu_gpu_bandwidth_mb_s)

    # list of accelerators
    jetson.accelerators.append("GPU")

    # max flops for every type of processor: needed for flop-based evaluation
    # FLOPS are computed using NVIDIA docs (for GPU) and https://en.wikipedia.org/wiki/FLOPS for CPU
    gpu_glops = 250 # 667: 667 is max, but GPU is never fully occupied due to registers limitation
    large_cpu_gflops = 16.28
    small_cpu_gflops = 10.85
    jetson.max_giga_flops_per_proc_type = [large_cpu_gflops, small_cpu_gflops, gpu_glops]

    return jetson

