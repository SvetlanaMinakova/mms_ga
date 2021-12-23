"""
Module analyzes DNNs and generates request for LUT-benchmarks
"""


def generate_bm_request(dnns, group_by_op=True):
    """
    Request benchmark for a set of DNNs
    :param dnns: set of DNNs
    :param group_by_op: group request by dnn operator, performed by the layer
    :return: benchmark for the set of DNNs
    """
    from eval.throughput.LUT_builder import LUTTree, build_lut_tree_from_jetson_benchmark, build_lut_tree
    from dnn_builders_zoo.keras_models_builder import build_keras_model
    from converters.keras_to_dnn import keras_to_dnn
    from dnn_builders_zoo.keras_models_builder import supported_keras_models
    dnns_lut = build_lut_tree(dnns)
    dnns_lut_as_table = dnns_lut.get_as_table()
    # dnns_lut.print_as_table()
    prev_op = "none"
    for record in dnns_lut_as_table:
        cur_op = record["op"]
        bm_config = generate_bm_layer_config(record)
        if cur_op != prev_op:
            print("")
            print("   ", "// operator: ", cur_op)
            print("    Config", bm_config)
        else:
            print("   ", bm_config)
        print("   ", "configs.push_back(config);")
        prev_op = cur_op


def generate_bm_layer_config(lut_record):
    """
    Generate config for evaluating a LUT record
    :param lut_record: record in LUT
    :return: config for evaluating a LUT record
    """
    # config example
    # {'op': 'conv', 'fs': 7, 'stride': 2, 'ifm': 3, 'ofm': 64, 'iw': 230, 'wpad': 0, 'hpad': 0, 'time': 0}
    # bm config example
    # conv
    # Config config = Config(ih, ch, neur, kh, hpad, stride);
    # matmul
    # Config config = Config(ih, ch, neur, kh);
    # nonlin
    # config =        Config(ih, ch, neur, kh, hpad, stride);

    if lut_record["op"] == "conv":
        bm_config = "config = Config(" + \
                    str(lut_record['iw']) + ", " + \
                    str(lut_record['ifm']) + ", " + \
                    str(lut_record['ofm']) + ", " + \
                    str(lut_record['fs']) + ", " + \
                    str(int(lut_record['hpad'])) + ", " + \
                    str(lut_record['stride']) + ");"
        return bm_config

    if lut_record["op"] in ["matmul", "gemm", "fc"]:
        bm_config = "config = Config(" + \
                    str(lut_record['iw']) + ", " + \
                    str(lut_record['ifm']) + ", " + \
                    str(lut_record['ofm']) + ", " + \
                    str(lut_record['fs']) + ");"
        return bm_config

    # default
    bm_config = "config = Config(" + \
                str(lut_record['iw']) + ", " + \
                str(lut_record['ifm']) + ", " + \
                str(lut_record['ofm']) + ", " + \
                str(lut_record['fs']) + ", " + \
                str(lut_record['hpad']) + ", " + \
                str(lut_record['stride']) + ");"

    return bm_config




