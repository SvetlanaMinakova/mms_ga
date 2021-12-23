
class EmbeddedPlatform:
    """
    Description of an embedded platform for power/perf evaluation
        parameters:
        input_examples bytes: bytes/ input_examples pixel or weight element
        total bandwidth: sum bandwidth over all input_examples transfer channels of the platform
        memories: list of platform memories (see Memory class)
        channels: list of input_examples transfer channels (see DataTransferChannel class)
    """
    def __init__(self, name):
        self.name = name

        # common properties here
        self.data_bytes = 0
        self.total_bandwidth = 0
        self.memory_partitions = 1
        self.idle_power = 0

        # sets
        self.processors = []
        self.channels = []
        self.memories = []

        # additional shared resources
        self.shared_resources = []

    ################################
    #   getters and setters    ###
    def get_processor(self, name):
        for proc in self.processors:
            if proc.name == name:
                return proc
        return None

    def get_memory(self, name):
        for memory in self.memories:
            if memory.name == name:
                return memory
        return None

    def get_channel(self, src, dst):
        for channel in self.channels:
            if channel.src == src and channel.dst == dst:
                return channel

    def get_resource(self, name):
        for resource in self.shared_resources:
            if resource.name ==name:
                return resource

    ################################
    #     print functions      ###

    def print_details(self, prefix=""):
        print(prefix, "name: ", self.name)
        print(prefix, "input_examples bytes: ", self.data_bytes)
        print(prefix, "total bandwidth (GB/s): ", self.total_bandwidth)
        print(prefix, "memory partitions: ", self.memory_partitions)
        print(prefix, "idle power (Watt): ", self.idle_power)
        lower_prefix = prefix + "  "
        lower_prefix_lvl2 = lower_prefix + "  "

        print(prefix, "memories: ")
        for memory in self.memories:
            print(lower_prefix, memory.name)
            memory.print_details(lower_prefix_lvl2)

        print(prefix, "channels: ")
        for channel in self.channels:
            print(lower_prefix, channel.src, "-->", channel.dst)
            channel.print_details(lower_prefix_lvl2)

        print(prefix, "additional shared resources: ")
        for resource in self.shared_resources:
            print(lower_prefix, resource.name)
            resource.print_details(lower_prefix_lvl2)

        print(prefix, "processors: ")
        for processor in self.processors:
            print(lower_prefix, processor.name)
            processor.print_details(lower_prefix_lvl2)


class Processor:
    """
    Embedded platform processor
        parameters:
        id: processor id (unque within processor type)
        type: processor type (CPU, GPU, TPU...)
        cores: number of cores (1 for one-core CPUs)
        max_perf: top processor performance in Gigaops/second
    """
    def __init__(self, proc_name, proc_type):
        # common loop-unrolling properties here
        self.name = proc_name
        self.type = proc_type
        self.parallel_comp_matrix = []
        self.max_perf = 0
        self.max_power = 0
        self.frequency = 0

        # computational model here
        self.unrolling = None

        # mapping here
        self.data_on_memory_mapping = {}

    ################################
    ###     print functions      ###
    def print_details(self, prefix=""):
        # print(prefix, "name: ", self.name)
        print(prefix, "type: ", self.type)
        print(prefix, "parallel comp. matrix: ", self.parallel_comp_matrix)
        print(prefix, "max. perf. (GOPS/s): ", self.max_perf)
        print(prefix, "max power (Watt): ", self.max_power)
        print(prefix, "frequency (GHz): ", self.frequency)

        lower_prefix = prefix + "  "
        print(prefix, "input_examples on memory mapping: ")
        print(lower_prefix, self.data_on_memory_mapping)
        print(prefix, "comp. model: ")
        self.unrolling.print_details(lower_prefix)

        # print(lower_prefix, self.unrolling)


class Memory:
    """
    Platform memory, utilized during DNN execution
    Parameters:
        size: total memory size (in bytes)
        bank_size: size of memory bank in bytes
        specific parameters: parameters, specific for this type of memory
    """
    def __init__(self, name, size, bank_size=0):
        self.name = name
        self.size = size
        self.bank_size = bank_size
        # self.special_parameters = {}

    ################################
    ###     print functions      ###
    def print_details(self, prefix=""):
        # print(prefix, "name: ", self.name)
        print(prefix, "size (bytes): ", self.size)
        print(prefix, "bank_size (bytes): ", self.bank_size)


class SharedResource:
    """
    Additional type of shared resource
    """
    def __init__(self, name, size):
        self.name = name
        self.size = size

    ################################
    ###     print functions      ###
    def print_details(self, prefix=""):
        # print(prefix, "name: ", self.name)
        print(prefix, "size (units): ", self.size)


class DataTransferChannel:
    """
    Data transfer channel : transfers i/o/weights input_examples during DNN execution
        Parameters:
        data_type: type of input_examples, transferred through the channel (i/o/weights)
        bandwidth: channel bandwidth GB/s
        width_bytes = input_examples transfer channel width in bytes ( =8 for 64-bit width)
        efficiency = efficiency of input_examples transfer channel, taking wasted computations into account
    """
    def __init__(self, src, dst, bandwidth):
        self.src = src
        self.dst = dst
        self.bandwidth = bandwidth
        self.width_bytes = 8 # 64-bit width
        # self.efficiency = 1

    ################################
    ###     print functions      ###
    def print_details(self, prefix=""):
        # print(prefix, "src: ", self.src)
        # print(prefix, "dst: ", self.dst)
        print(prefix, "bandwidth (GB/s): ", self.bandwidth)
        print(prefix, "width bytes: ", self.width_bytes)










