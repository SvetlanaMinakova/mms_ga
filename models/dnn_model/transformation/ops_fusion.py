from models.dnn_model.dnn import DNN


def fuse_built_in(dnn: DNN):
    """
    Fuse all operators marked as built-in
    :param dnn: dnn model
    :return: dnn model with fused operators
    """
    layers_to_fuse = [layer for layer in dnn.get_layers() if layer.built_in]

    for layer in layers_to_fuse:
        fuse_layer(dnn, layer, verbose=True)

    dnn.sort_connections_in_layers_order()


def fuse_layer(dnn, layer, verbose):
    """
    Fuse layer to the parent layer in the DNN
    :param dnn: dnn
    :param layer: layer to fuse
    :param verbose: print details
    """
    # find input connections of the layer
    input_connections = dnn.get_layer_input_connections(layer)
    inputs_num = len(input_connections)
    output_connections = dnn.get_layer_output_connections(layer)
    outputs = [output_connection.dst for output_connection in output_connections]

    # if layer has a single input connection, it is fused with input
    if inputs_num == 1:
        single_input_connection = input_connections[0]
        # parent layer: layer to which the current layer will be fused
        parent = single_input_connection.src
        fuse_layer_to_prev_layer(layer, parent)
        # remove current layer and its I/O connections from the dnn
        dnn.remove_layer(layer)

        # connect parent to the outputs of the layer-to-be-fused
        # THIS IS ALREADY DONE IN dnn.remove_layer()
        # for output in outputs:
        #    dnn.connect_layers_by_name(parent.name, output.name)

    if inputs_num == 0:
        for output_layer in outputs:
            fuse_layer_to_next_layer(layer, output_layer)

        # remove current layer and its I/O connections from the dnn
        dnn.remove_layer(layer)

    # layer cannot be incapsulated if it has multiple (>1) input connections
    if inputs_num > 1:
        if verbose:
            print("WARNING: incapsulation of layer ", layer.name,
                  "skipped. Layer is expected to have <= 1 input, but ", inputs_num, "inputs are registered")


def fuse_layer_to_prev_layer(layer, prev_layer):
    """
    Fuse layer to the previous (parent) layer
    :param layer: layer which will be fused and deleted from the DNN
    :param prev_layer: layer which will embed the fused layer
    :param verbose: print details
    """

    # add all subops of current layer to the parent layer
    for subop in layer.fused_ops:
        if subop not in prev_layer.fused_ops:
            prev_layer.fuse_subop(subop)
    prev_layer.fuse_subop(layer.subop)

    # change parent output format
    prev_layer.ofm = layer.ofm
    prev_layer.oh = layer.oh
    prev_layer.ow = layer.ow




def fuse_layer_to_next_layer(layer, next_layer):
    """
    Fuse layer to the next layer
    :param layer: layer which will be fused and deleted from the DNN
    :param next_layer: layer which will embed the fused layer
    :param verbose: print details
    """

    # add all subops of current layer to the next layer
    for subop in layer.fused_ops:
        if subop not in next_layer.fused_ops:
            next_layer.fuse_subop(subop)
    next_layer.fuse_subop(layer.subop)

    # change parent input format
    """
    next_layer.ifm = layer.ifm
    next_layer.ih = layer.ih
    next_layer.iw = layer.iw
    """









