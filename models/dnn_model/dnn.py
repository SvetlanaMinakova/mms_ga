"""
This module contains model of a deep neural network (DNN)
"""


class DNN:
    """
    Deep Neural Network
    Attributes:
        name: (str) dnn name
        __layers: dnn layers. Every layer is defined as an object of class Layer (defined below).
         DNN layers represent DNN functionality.
        __connections: connections between DNN edges. Every connection is defined as
         an object of class Connection (defined below). DNN edges represent data dependencies within the DNN.
        __inputs and __outputs: external I/Os providing/consuming data to/from DNN layers.
        Every input/output is represented as an object of ExternalInputConnection and ExternalOutputConnection class
        (defined below), respectively
        __next_layer_id: (int) id of next layer to be added into the DNN
    """
    def __init__(self, name="DNN"):
        self.name = name
        self.__layers = []
        self.__connections = []

        self.__inputs = []
        self.__outputs = []

        self.__next_layer_id = 0

    # equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def get_layer_input_connections(self, layer):
        layer_input_connections = []
        for connection in self.__connections:
            if connection.dst == layer:
                layer_input_connections.append(connection)
        return layer_input_connections

    def get_layer_output_connections(self, layer):
        layer_output_connections = []
        for connection in self.__connections:
            if connection.src == layer:
                layer_output_connections.append(connection)
        return layer_output_connections

    def get_layer_id(self, layer):
        l_id = 0
        for dnn_layer in self.__layers:
            if dnn_layer == layer:
                return l_id
            l_id = l_id + 1
        return -1

    def get_connection_id(self, connection):
        c_id = 0
        for dnn_connection in self.__connections:
            if dnn_connection == connection:
                return c_id
            c_id = c_id + 1
        return -1

    def find_layer_by_name(self, name):
        for layer in self.__layers:
            if layer.name == name:
                return layer
        return None

    def get_input_layer(self):
        if not self.__layers:
            return None
        return self.__layers[0]

    def get_output_layer(self):
        if not self.__layers:
            return None
        return self.__layers[-1]

    def add_layer(self, layer):
        layer.id = self.__next_layer_id
        self.__layers.append(layer)
        self.__next_layer_id = self.__next_layer_id + 1

    def insert_layer(self, layer, pos):
        self.__layers.insert(pos, layer)
        self.re_index_layers()

    def remove_layer(self, layer):
        # get I/O connections
        input_connections = self.get_layer_input_connections(layer)
        output_connections = self.get_layer_output_connections(layer)

        # re-purpose connections: directly connect every layer input_examples with every layer output
        for input_connection in input_connections:
            for output_connection in output_connections:
                new_src = input_connection.src
                new_dst = output_connection.dst
                self.connect_layers_by_name(new_src.name, new_dst.name)

        # remove (old) input connections
        for connection in input_connections:
            self.__connections.remove(connection)

        # remove (old) output connections
        for connection in output_connections:
            self.__connections.remove(connection)

        # remove layer
        self.__layers.remove(layer)
        # change layer ids
        self.re_index_layers()

    def re_index_layers(self):
        lid = 0
        for layer in self.__layers:
            layer.id = lid
            lid += 1
        self.__next_layer_id = lid

    def add_connection(self, connection):
        self.__connections.append(connection)

    def connect_layers(self, src_id, dst_id):
        src_layer = self.__layers[src_id]
        dst_layer = self.__layers[dst_id]
        connection = Connection(src_layer, dst_layer)
        self.__connections.append(connection)

    def connect_layers_by_name(self, src_name, dst_name):
        src_layer = self.find_layer_by_name(src_name)
        dst_layer = self.find_layer_by_name(dst_name)
        if src_layer is None:
            raise Exception("DNN connection addition error: src layer " + src_name + " not found!")
        if dst_layer is None:
            raise Exception("DNN connection addition error: dst layer " + dst_name + " not found!")

        existing_connection = self.find_connection_by_layers(src_layer, dst_layer)
        if existing_connection is not None:
            print("Connection", existing_connection, " not added, because it already exists!")

        connection = Connection(src_layer, dst_layer)
        self.__connections.append(connection)
        # print("connection created: ", connection)
        # print("src:", connection.src, ",dst: ", connection.dst)

    def stack_layer(self, layer):
        """
        Add layer and chain-connect it to the current dnn_model output layer
        :param layer: new layer
        """
        self.add_layer(layer)
        if len(self.__layers) > 1:
            src = self.__layers[-2]
            dst = self.__layers[-1]
            connection = Connection(src, dst)
            self.add_connection(connection)

    def sort_connections_in_layers_order(self):
        """ Sort connections by layers order"""
        sorted_connections = []
        for layer in self.__layers:
            layer_output_connections = self.get_layer_output_connections(layer)
            for connection in layer_output_connections:
                sorted_connections.append(connection)
        self.__connections = sorted_connections

    def get_layers(self):
        return self.__layers

    def get_connections(self):
        return self.__connections

    def find_connection_by_layers(self, src, dst):
        for connection in self.__connections:
            if connection.src == src and connection.dst == dst:
                return connection

    # give short description of the dnn_model
    def __str__(self):
        return "{name: " + self.name + ", layers: " + str(len(self.__layers)) + ", connections: " + str(len(self.__connections)) + "}"

    # print full description of the dnn_model
    def print_details(self, print_layers=True, print_connections=True, print_ios=True):
        print(self)
        if print_layers:
            print("Layers: ")
            for layer in self.__layers:
                print("   ", layer)
        if print_connections:
            print("Connections: ")
            for connection in self.__connections:
                print("   ", connection)
        if print_ios:
            print("Inputs: ")
            for connection in self.__inputs:
                print("   ", connection)
            print("Outputs: ")
            for connection in self.__outputs:
                print("   ", connection)

    def set_auto_unique_layer_names(self):
        """
        Set automatically generated unique names for every dnn layer
        """
        for layer in self.__layers:
            layer.name = layer.op + str(layer.id)

    """
    External data sources and consumers
    """

    def set_auto_ios(self):
        """
        Set automatically external inputs and outputs for the DNN
        """
        self.__inputs = []
        self.__outputs = []
        input_layer = self.get_input_layer()
        self.add_external_input("input_data", input_layer.iw, input_layer.ih, input_layer.ifm)
        output_layer = self.get_output_layer()
        self.add_external_output("output_data", output_layer.ow, output_layer.oh, output_layer.ofm)

    def add_external_input(self, name, iw, ih, ifm, dnn_layer=None):
        """
        Add new external input
        :param name: name of external input
        :param iw: input width
        :param ih: input height
        :param ifm: input feature maps
        :param dnn_layer: layer in the dnn that accepts this input.
        If unspecified, first layer in the dnn layers list is selected as dnn_layer
        """
        input_layer = self.get_input_layer() if dnn_layer is None else dnn_layer
        external_input = ExternalInputConnection(name, iw, ih, ifm, input_layer)
        self.__inputs.append(external_input)

    def add_external_output(self, name, ow, oh, ofm, dnn_layer=None):
        """
        Add new external output
        :param name: name of external input
        :param ow: output width
        :param oh: output height
        :param ofm: output feature maps
        :param dnn_layer: layer in the dnn that produces data into this output.
        If unspecified, last layer in the dnn layers list is selected as dnn_layer
        """
        output_layer = self.get_output_layer() if dnn_layer is None else dnn_layer
        external_output = ExternalOutputConnection(name, ow, oh, ofm, output_layer)
        self.__outputs.append(external_output)

    def remove_external_input(self, external_input):
        self.__inputs.remove(external_input)

    def clean_external_inputs(self):
        # remove all external inputs
        self.__inputs = []

    def remove_external_output(self, external_output):
        self.__outputs.remove(external_output)

    def clean_external_outputs(self):
        # remove all external outputs
        self.__outputs = []

    def get_layer_external_inputs(self, layer):
        layer_external_inputs = []
        for inp in self.__inputs:
            if inp.dnn_layer == layer:
                layer_external_inputs.append(inp)
        return layer_external_inputs

    def get_inputs(self):
        return self.__inputs

    def get_outputs(self):
        return self.__outputs


def set_built_in(dnn, ops):
    """
    Set built-in (fused) operations within a dnn.
    The layers, performing a built-in operation are fused with previous layers and
    do not have an intermediate computational result (output buffer) of their own
    :param dnn: dnn
    :param ops: list of names of built-in operators, e.g. ["relu", "bn"]
    """
    for layer in dnn.get_layers():
        if layer.op in ops:
            layer.built_in = True


class Layer:
    """
    A CNN layer
    """
    def __init__(self, res, op, fs, ifm, ofm, bordermode):
        # operator performed by the layer
        self.op = op
        # sub-type of operator, performed by the layer
        # e.g. maxpool for pool op, or separable_conv for conv op
        self.subop = self.op
        # kernel (filter) size
        self.fs = fs
        # stride
        self.stride = 1
        # input_examples feature maps
        self.ifm = ifm
        # output feature maps
        self.ofm = ofm
        # input_examples image resolution (input_examples image size)
        self.ih = self.iw = self.res = res
        # output image size (default)
        self.ow = self.oh = res
        # unique layer id within DNN. Set by DNN,
        # when layer is added to the DNN topology.
        self.id = id
        # layer name: is NOT necessarily unique, it's just a label
        self.name = op
        # borders processing: only valid for convolutional /pooling layers
        self.__bordermode = bordermode
        self.set_autopads()
        # padding
        self.pads = [0, 0, 0, 0]
        self.time_eval = 0
        # if layer is built in the previous layer
        # useful for ops such as BatchNormalization or ReLU
        self.built_in = False
        self.phases = 1
        self.fused_ops = []

    # equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    """
    getters and setters
    """

    """
    Data load computation/extraction formulas
    """
    def get_dim_names(self, data_type):
        if data_type == "input_data":
            return self.__get_inp_dim_names()
        if data_type == "output_data":
            return self.__get_outp_dim_names()
        if data_type == "weights":
            return self.__get_weight_dim_names()
        return []

    def __get_weight_dim_names(self):
        if self.op == "conv":
            return ["ifm", "ofm", "kh", "kw"]
        if self.op in ["gemm", "fc", "matmul"]:
            # NOTE: tied to gemm ops tensor
            return ["oh", "ow"]
        return []

    def __get_outp_dim_names(self):
        # NOTE: tied to gemm ops tensor
        if self.op in ["gemm", "fc", "matmul"]:
            return ["ow"]
        return ["ofm", "oh", "ow"]

    def __get_inp_dim_names(self):
        # NOTE: tied to gemm ops tensor
        if self.op in ["gemm", "fc", "matmul"]:
            return ["oh"]
        if self.op == "pool":
            return ["oh", "ow", "ofm", "kh", "kw"]
        return ["oh", "ow", "ifm"]

    """
    Printout
    """

    def __str__(self):
        return "{id:" + str(self.id) + ", name: " + self.name +\
               ", op: " + self.op + ", subop: " + self.subop + \
               ", fused_ops:" + str(self.fused_ops) + \
               ", fs: " + str(self.fs) + \
               ", ifm: " + str(self.ifm) + \
               ", iw: " + str(self.iw) + \
               ", ih: " + str(self.ih) + \
               ", ofm: " + str(self.ofm) + \
               ", ow: " + str(self.ow) + \
               ", oh: " + str(self.oh) + "}"

    """
    Pads processing
    """
    def set_border_mode(self, bordermode: str):
        if bordermode not in ["same", "full", "valid"]:
            raise Exception("Border mode setting error, mode " + bordermode +
                            " is unsupported. Please choose from [same, full, valid]")
        self.__bordermode = bordermode
        self.set_autopads()

    def set_autopads(self):
        self.pads = self.__get_autopads()

    def set_pads(self, wpad, hpad):
        self.pads = [wpad, wpad, hpad, hpad]

    def __get_autopads(self):
        if self.__bordermode == "same" and self.op in ["conv", "pool"]:
            hpad = wpad = ((self.res - 1) * self.stride - self.res + self.fs)/2
            return [wpad, wpad, hpad, hpad]

        # default = valid
        return [0, 0, 0, 0]

    def get_hpad(self):
        return self.pads[1]

    def get_wpad(self):
        return self.pads[0]

    """
    Operators fusion
    """
    def fuse_subop(self, subop):
        self.fused_ops.append(subop)

    def has_fused(self):
        return len(self.fused_ops) > 0


class Connection:
    """
    A Connection between two layers of a DNN
    """
    def __init__(self, src: Layer, dst: Layer):
        self.src = src
        self.dst = dst
        self.visited = False
        self.double_buffer = False

    # equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __str__(self):
        return "{src: " + str(self.src.id) + ", dst: " + str(self.dst.id) + "}"


class ExternalInputConnection:
    """
    A Connection between a layer of a DNN and an external source of data
    consumed by the DNN
    """
    def __init__(self, name, data_w, data_h, data_ch, dnn_layer: Layer):
        self.data_layer = Layer(data_w, op="data", fs=1, ifm=data_ch, ofm=data_ch, bordermode="same")
        self.data_layer.iw = self.data_layer.ow = data_w
        self.data_layer.ih = self.data_layer.oh = data_h
        self.data_layer.name = name
        self.data_layer.subop = "input"
        self.data_layer.id = -1

        self.dnn_layer = dnn_layer

        self.visited = False
        self.double_buffer = False

    # equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __str__(self):
        return "{src (external): " + str(self.data_layer) + ", dst: " + str(self.dnn_layer.id) + "}"


class ExternalOutputConnection:
    """
    A Connection between a layer of a DNN and an external consumer of data
    produced by the DNN
    """
    def __init__(self, name, data_w, data_h, data_ch, dnn_layer: Layer):
        self.data_layer = Layer(data_w, op="data", fs=1, ifm=data_ch, ofm=data_ch, bordermode="same")
        self.data_layer.iw = self.data_layer.ow = data_w
        self.data_layer.ih = self.data_layer.oh = data_h
        self.data_layer.name = name
        self.data_layer.subop = "output"
        self.data_layer.id = -1

        self.dnn_layer = dnn_layer

        self.visited = False
        self.double_buffer = False

    # equality
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __str__(self):
        return "{src: " + str(self.dnn_layer.id) + ", dst (external): " + str(self.data_layer) + "}"


def layer_has_null_or_empty_pads(layer):
    """
    Check if layer has null or empty pads
    :param layer: layer
    :return: True if layer has null or empty pads
    """
    if not layer.pads:
        return True
    for elem in layer.pads:
        if elem > 0:
            return False
    return True