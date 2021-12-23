class CSDFGraph:
    def __init__(self, name):
        self.name = name
        self.__channels = []
        self.__actors = []

    def add_actor(self, actor):
        self.__actors.append(actor)

    def connect_actors_by_ids(self, src_id, dst_id, prod_seq, cons_seq):
        src_actor = self.__actors[src_id]
        dst_actor = self.__actors[dst_id]
        channel = CSDFFIFOChannel(src_actor, dst_actor, prod_seq, cons_seq)
        self.__channels.append(channel)

    def get_input_channels(self, actor):
        channels = []
        for channel in self.__channels:
            if channel.dst == actor:
                channels.append(channel)
        return channels

    def get_output_channels(self, actor):
        channels = []
        for channel in self.__channels:
            if channel.src == actor:
                channels.append(channel)
        return channels

    def get_actors(self):
        return self.__actors

    def get_channels(self):
        return self.__channels

    def __str__(self):
        return "{name: " + self.name + ", actors: " + str(len(self.__actors)) + \
               ", FIFO channels: " + str(len(self.__channels)) + "}"

    def print_details(self, print_actors=True, print_channels=True):
        print(self)
        if print_actors:
            print("Actors: ")
            for actor in self.__actors:
                print("   ", actor)
        if print_channels:
            print("FIFO channels: ")
            for channel in self.__channels:
                print("   ", channel)


class CSDFFIFOChannel:
    def __init__(self, src, dst, prod_seq, cons_seq):
        self.src = src
        self.dst = dst
        self.prod_seq = prod_seq
        self.cons_seq = cons_seq

    def __str__(self):
        return "{src: " + str(self.src.name) + \
               ", dst: " + str(self.dst.name) + \
               ", prod: " + rate_seq_to_short_str(self.prod_seq) + \
               ", cons: " + rate_seq_to_short_str(self.cons_seq) + \
               "}"


class CSDFActor:
    def __init__(self, name, exec_seq):
        self.name = name
        self.exec_seq = exec_seq
        self.phases = len(exec_seq)
        self.time_per_phase = [0 for _ in range(self.phases)]

    def __str__(self):
        return "{name: " + self.name +\
               ", phases: " + str(self.phases) +\
               ", exec_seq:" + exec_seq_to_short_str(self.exec_seq)+ \
               ", exec_time:" + rate_seq_to_short_str(self.time_per_phase) + \
               "}"


def exec_seq_to_short_str(exec_seq:[]):
    if not exec_seq:
        return "[]"

    short_str = "["

    prev_func = exec_seq[0]
    func_repeated = 0
    for func in exec_seq:
        if func == prev_func:
            func_repeated += 1
        else:
            short_str += str(func_repeated) + "*" + prev_func + ";"
            func_repeated = 1
        prev_func = func

    if func_repeated > 0:
        short_str += str(func_repeated) + "*" + prev_func + ";"

    short_str += "]"
    return short_str


def rate_seq_to_short_str(rate_seq:[]):
    if not rate_seq:
        return "[]"

    short_str = "["

    prev_rate = rate_seq[0]
    rate_repeated = 0
    for rate in rate_seq:
        if rate == prev_rate:
            rate_repeated += 1
        else:
            short_str += str(rate_repeated) + "*" + str(prev_rate) + ";"
            rate_repeated = 1
        prev_rate = rate

    if rate_repeated > 0:
        short_str += str(rate_repeated) + "*" + str(prev_rate) + ";"

    short_str += "]"
    return short_str


def check_csdfg_consistency(csdfg: CSDFGraph, verbose=True):
    """
    Check consistency of the csdfg graph
    :param csdfg: CSDF graph
    :return: True if csdfg is consistent and False otherwise
    """
    for channel in csdfg.get_channels():
        if sum(channel.prod_seq) != sum(channel.cons_seq):
            if verbose:
                print("CSDF graph", csdfg.name, "is inconsistent: FIFO channel", channel, "prod rate",
                      sum(channel.prod_seq), "!= cons rate", sum(channel.cons_seq))
            return False
    return True



