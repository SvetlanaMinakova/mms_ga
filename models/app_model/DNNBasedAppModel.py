from high_throughput.scheduling.dnn_scheduling import DNNScheduling
"""
DNN-based application model
"""


class SingleDNNAppModel:
    def __init__(self, name, dnn, partitioning, mapping, scheduling: DNNScheduling, buffers):
        self.name = name
        self.dnn = dnn
        self.partitioning = partitioning
        self.mapping = mapping
        self.scheduling = scheduling
        self.buffers = buffers


class MultiDNNAppModel:
    def __init__(self, name, dnns, partitioning, mapping, scheduling: DNNScheduling, buffers):
        self.name = name
        self.dnns = dnns
        self.partitioning = partitioning
        self.mapping = mapping
        self.scheduling = scheduling
        self.buffers = buffers


class InterDNNConnection:
    """
        Connection between two DNNs/ DNN partitions
        Attributes:
            name: connection name
            src (DNN), dst (DNN): source and destination, represented as (analytical) DNN models
            src_output_name, dst_output_name: names of external output/ external input_examples in the src and dst DNN,
            corresponding to this  InterPartitionsConnection
            data_w, data_h, data_ch: dimensions of data tensor, transferred through connection:
            width, height and number of channels, respectively
    """
    def __init__(self, name, src, dst, data_w, data_h, data_ch):
        self.name = name
        self.src = src
        self.dst = dst
        self.data_w = data_w
        self.data_h = data_h
        self.data_ch = data_ch

    def __str__(self):
        return "{name: " + self.name + ", src: " + self.src.name + ", dst: " + self.dst.name + \
               ",data: [w: " + str(self.data_w) + ", h: " + str(self.data_h) + ", c: " + str(self.data_ch) + "]" "}"

    def get_data_size(self):
        return self.data_w * self.data_h * self.data_ch

