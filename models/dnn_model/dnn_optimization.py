from models.dnn_model.dnn import DNN, Layer


def remove_skip_layers(dnn):
    """ Remove all layers with skip-operator"""
    for layer in dnn.get_layers():
        if layer.op == "skip":
            dnn.remove_layer(layer)
