from models.csdf_model.csdf import CSDFGraph
from models.dnn_model.dnn import DNN
from eval.memory.dnn_model_mem_eval import eval_connection_buffer_tokens

""" Buffers store data, associated with applications"""


class DataBuffer:
    """ Buffer, that stores application data"""
    def __init__(self, name, size):
        self.name = name
        self.size = size
        self.users = []

    def assign(self, user):
        self.users.append(user)

    def __str__(self):
        return "{name: " + self.name + ", size: " + str(self.size) + ", users num: " + str(len(self.users)) + "}"

    def print_details(self, print_users=True):
        print(self)
        if print_users:
            print("users: ")
            for user in self.users:
                print(" ", user)


class CSDFGDataBuffer(DataBuffer):
    """ Buffer, that stores data of an application, represented as CSDF Graph"""
    def __init__(self, name, size):
        super(CSDFGDataBuffer, self).__init__(name, size)
        # FIFO channels, using the CSDFG buffer
        self.channels = []

    def __str__(self):
        return "{name: " + self.name + ", size: " + str(self.size) + ", channels num: " + str(len(self.channels)) + "}"

    def print_details(self, print_users=True):
        print(self)
        if print_users:
            print("channels: ")
            for channel in self.channels:
                print(" ", channel)


def build_naive_csdfg_buffers(csdfg: CSDFGraph):
    """
    Build naive CSDF graph buffers, where every CSDF actor is allocated it's own buffer
    :param csdfg: CSDF graph
    :return: list of naive CSDF graph buffers,
    """
    csdfg_buf = []
    for channel in csdfg.get_channels():
        buf_name = channel.src.name + "_" + channel.dst.name
        max_prod_rate = sum(channel.prod_seq)
        max_cons_rate = sum(channel.cons_seq)
        buf_size = max(max_prod_rate, max_cons_rate)
        buf = CSDFGDataBuffer(buf_name, buf_size)
        buf.channels.append(channel)
        csdfg_buf.append(buf)
    return csdfg_buf


class DNNDataBuffer(DataBuffer):
    """
    A buffer, used to store data of a (multi-) DNN-based application
    Attributes:
        connections_per_dnn:  dict where key = DNN, represented as the (analytical) dnn model,
        value = list of dnn connections, storing data in this buffer
    """
    def __init__(self, name, size):
        super(DNNDataBuffer, self).__init__(name, size)

        self.connections_per_dnn = {}

    def add_connection(self, dnn, connection):
        if dnn in self.connections_per_dnn.keys():
            values = self.connections_per_dnn[dnn]
            values.append(connection)
            self.connections_per_dnn[dnn] = values
        else:
            self.connections_per_dnn[dnn] = [connection]
        # update size
        connection_tokens = eval_connection_buffer_tokens(connection)
        if connection_tokens > self.size:
            self.size = connection_tokens

    def is_empty(self):
        return len(self.connections_per_dnn) == 0

    def print_details(self, print_stored_connections=True):
        print("DNN buffer: { size: ", self.size, "tokens }")
        if print_stored_connections:
            print("  - stored connections: ")
            for dnn in self.connections_per_dnn.keys():
                print("    - dnn: ", dnn)
                for connection in self.connections_per_dnn[dnn]:
                    print("      -", connection.src.name, "-->", connection.dst.name,
                          " (", eval_connection_buffer_tokens(connection), "tokens )")


def build_naive_dnn_buffers(dnns_list):
    """
    Build a set of naive (non-reused) buffers for list of dnns
    :param dnns_list: list of dnns
    :return:  set of naive buffers for list of dnns
    """
    buffers = []
    for dnn in dnns_list:
        for connection in dnn.get_connections():
            naive_buffer = DNNDataBuffer(("B" + str(len(buffers))), eval_connection_buffer_tokens(connection))
            buffers.append(naive_buffer)
            naive_buffer.add_connection(dnn, connection)
    return buffers
