from models.dnn_model.dnn import DNN, Layer, ExternalInputConnection, ExternalOutputConnection
import copy


def external_ios_to_data_layers(dnn: DNN):
    """
    In a DNN, transform all the external inputs and outputs, defined as objects of
    ExternalInputConnection and ExternalOutputConnection classes, respectively
    into data layers
    :param dnn: DNN: dnn to perform transformations in
    """
    # build-in inputs
    inputs = dnn.get_inputs()
    for dnn_input in inputs:
        inp_layer = copy.deepcopy(dnn_input.data_layer)
        inp_layer.subop = "input"

        # insert external input represented as a layer
        # into the beginning of dnn layers list
        layer_insert_pos = 0 # dnn_input.dnn_layer.id
        dnn.insert_layer(inp_layer, layer_insert_pos)

        # connect external input represented as a layer to
        # the data-receiving layer in the dnn
        dnn.connect_layers_by_name(inp_layer.name, dnn_input.dnn_layer.name)

    # remove all external inputs
    dnn.clean_external_inputs()

    # build-in outputs
    outputs = dnn.get_outputs()
    for dnn_output in outputs:
        outp_layer = copy.deepcopy(dnn_output.data_layer)
        outp_layer.subop = "output"

        # append external input represented as a layer
        # at the end of dnn layers list
        dnn.add_layer(outp_layer)

        # connect external output represented as a layer to
        # the data-producing layer in the dnn
        dnn.connect_layers_by_name(dnn_output.dnn_layer.name, outp_layer.name)

    # remove all external outputs
    dnn.clean_external_outputs()

    dnn.sort_connections_in_layers_order()


def data_layers_to_external_ios(dnn: DNN):
    """
    In a DNN, transform all inpout and output data layers into
    external inputs and outputs, defined as objects of
    ExternalInputConnection and ExternalOutputConnection classes, respectively
    :param dnn: DNN: dnn to perform transformations in
    """
    input_layers, output_layers = get_input_and_output_data_layers(dnn)
    # transform input layers into external input connections
    for input_layer in input_layers:
        layer_output_connections = dnn.get_layer_output_connections(input_layer)
        inp_id = 0
        for connection in layer_output_connections:
            input_name = input_layer.name if len(layer_output_connections) < 2 else input_layer.name + str(inp_id)
            dnn.add_external_input(input_name, input_layer.iw, input_layer.ih, input_layer.ifm, connection.dst)
            inp_id += 1

    for input_layer in input_layers:
        dnn.remove_layer(input_layer)

    # transform output layers into external output connections
    for output_layer in output_layers:
        layer_input_connections = dnn.get_layer_input_connections(output_layer)
        outp_id = 0
        for connection in layer_input_connections:
            output_name = output_layer.name if len(layer_input_connections) < 2 else output_layer.name + str(outp_id)
            dnn.add_external_output(output_name, output_layer.ow, output_layer.oh, output_layer.ofm, connection.src)
            outp_id += 1

    for output_layer in output_layers:
        dnn.remove_layer(output_layer)


def get_input_and_output_data_layers(dnn: DNN):
    """
    Get lists of input and output data layers (each represented as an object of class Layer)
    :param dnn: DNN
    :return: input_layers:[Layer], output_layers[Layer]
    """
    # all data layers
    data_layers = [layer for layer in dnn.get_layers() if layer.op == "data"]

    # sort layers into input and output layers
    # sort by sub-operator
    input_layers = [data_layer for data_layer in data_layers if data_layer.subop == "input"]
    output_layers = [data_layer for data_layer in data_layers if data_layer.subop == "output"]

    sorted_layers_num = len(input_layers) + len(output_layers)
    if sorted_layers_num < len(data_layers):
        # process unsorted layers
        # sort by connections number
        for data_layer in data_layers:
            if data_layer not in input_layers:
                if data_layer not in output_layers:
                    inputs_num = len(dnn.get_layer_input_connections(data_layer))
                    if inputs_num == 0:
                        data_layer.subop = "input"
                        input_layers.append(data_layer)
                    else:
                        data_layer.subop = "output"
                        output_layers.append(data_layer)

    return input_layers, output_layers








