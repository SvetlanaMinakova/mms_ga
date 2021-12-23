from models.dnn_model.dnn import DNN, Layer
import json
from converters.json_converters.json_util import extract_or_default
from types import SimpleNamespace


def parse_json_dnn(path):
    """
    Converts a JSON File into an (analytical) DNN model
    :param path: path to .json file
    """
    with open(path, 'r') as file:
        if file is None:
            raise FileNotFoundError
        else:
            json_dnn = json.load(file)
            dnn_name = extract_or_default(json_dnn, "name", "DNN")

            dnn = DNN(dnn_name)

            # parse layers
            json_layers = extract_or_default(json_dnn, "_DNN__layers", [])
            for json_layer in json_layers:
                layer = parse_json_layer(json_layer)
                dnn.add_layer(layer)

            # parse connections
            json_connections = extract_or_default(json_dnn, "_DNN__connections", [])
            for connection in json_connections:
                parse_json_connection(dnn, connection)

            # parse external inputs
            external_inputs = extract_or_default(json_dnn, "_DNN__inputs", [])
            for json_external_input in external_inputs:
                parse_json_external_input(dnn, json_external_input)

            # parse external outputs
            external_outputs = extract_or_default(json_dnn, "_DNN__outputs", [])
            for json_external_output in external_outputs:
                parse_json_external_output(dnn, json_external_output)

    return dnn


def parse_json_layer(json_layer):
    """
    JSON description of DNN layer
    :param json_layer: DNN layer in JSON format
    :return: dnn layer represented as an object of Layer class
    """
    layer = Layer(json_layer["res"],
                  json_layer["op"],
                  json_layer["fs"],
                  json_layer["ifm"],
                  json_layer["ofm"],
                  json_layer["_Layer__bordermode"])

    # operator details
    layer.subop = extract_or_default(json_layer, "subop", layer.op)
    layer.fused_ops = extract_or_default(json_layer, "fused_ops", [])

    # stride and padding
    layer.pads = extract_or_default(json_layer, "pads", layer.pads)
    layer.stride = extract_or_default(json_layer, "stride", layer.stride)

    # extra data formats
    layer.ih = extract_or_default(json_layer, "ih", layer.ih)
    layer.iw = extract_or_default(json_layer, "iw", layer.iw)
    layer.oh = extract_or_default(json_layer, "oh", layer.oh)
    layer.ow = extract_or_default(json_layer, "ow", layer.ow)

    # additional
    layer.name = extract_or_default(json_layer, "name", layer.name)
    layer.phases = extract_or_default(json_layer, "phases", layer.phases)
    layer.time_eval = extract_or_default(json_layer, "time_eval", layer.time_eval)
    layer.built_in = extract_or_default(json_layer, "built_in", layer.built_in)

    return layer


def parse_json_connection(dnn: DNN, json_connection):
    src_id = json_connection["src"]
    dst_id = json_connection["dst"]
    dnn.connect_layers(src_id, dst_id)


def parse_json_external_input(dnn: DNN, json_external_input):
    # determine dnn layer to which the input is attached
    dnn_layer_id = json_external_input["dnn_layer"]
    dnn_layer = dnn.get_layers()[dnn_layer_id]

    # determine data layer
    json_data_layer = json_external_input["data_layer"]
    data_layer = parse_json_layer(json_data_layer)

    # add input
    dnn.add_external_input(data_layer.name, data_layer.ow, data_layer.oh, data_layer.ofm, dnn_layer)


def parse_json_external_output(dnn: DNN, json_external_output):
    # determine dnn layer to which the output is attached
    dnn_layer_id = json_external_output["dnn_layer"]
    dnn_layer = dnn.get_layers()[dnn_layer_id]

    # determine data layer
    json_data_layer = json_external_output["data_layer"]
    data_layer = parse_json_layer(json_data_layer)

    # add output
    dnn.add_external_output(data_layer.name, data_layer.iw, data_layer.ih, data_layer.ifm, dnn_layer)

