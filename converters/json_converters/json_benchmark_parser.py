from models.dnn_model.dnn import DNN, Layer
from converters.json_converters.json_util import parse_list


def parse_jetson_bm_as_annotated_dnn(filepath):
    dnn = DNN()
    layer_bms_list = parse_list(filepath)
    for layer_bm in layer_bms_list:
        layer = parse_jetson_layer_bm_record(layer_bm)
        dnn.add_layer(layer)
    return dnn


def parse_jetson_bm_as_layers_list(filepath):
    layers = []
    layer_bms = parse_list(filepath)
    for layer_bm in layer_bms:
        layer = parse_jetson_layer_bm_record(layer_bm)
        layers.append(layer)
    return layers


def parse_jetson_layer_bm_record(layer_desc):
    """
    Parse layer
    :param layer_desc json_converters description of a CNN layer
    :return DNN layer as a class
    """
    op = layer_desc["op"]
    ifm = layer_desc["ch"]
    fs = layer_desc["kh"]
    ofm = layer_desc["ofm"]
    res = layer_desc["iw"]

    layer = Layer(res=res, op=op, fs=fs, ifm=ifm, ofm=ofm, bordermode="valid")

    layer.time_eval = layer_desc["time"]
    layer.oh = layer.ow = layer_desc["oh"]
    layer.ih = layer.iw = layer.res

    # set stride
    if "stride" not in layer_desc:
        layer.stride = 1
    else:
        layer.stride = layer_desc["stride"]

    # get border mode/pads
    if "hpad" not in layer_desc or "wpad" not in layer_desc:
        if layer.oh == layer.res:
            layer.set_border_mode("same")
            layer.set_autopads()
        hpad = int(((layer.oh - 1) * layer.stride - layer.ih + layer.fs)/2)
        wpad = int(((layer.ow - 1) * layer.stride - layer.iw + layer.fs) / 2)
        layer.set_pads(wpad, hpad)

    else:
        layer.set_pads(layer_desc["wpad"], layer_desc["hpad"])

    return layer