from eval.latency.lut.LUT_builder import LUTTree


def estim_layer_lut(layer, lut: LUTTree):
    """
    Estimate layer performance, using lookup-tables (LUT)
    :param layer: layer
    :param lut: lookup table, built from measurements, performed
    on the platform
    :return:  layer performance, estimaeted using lookup-tables (LUT)
    """
    # print("search layer ", layer, "in lut")
    eval_lut = 0
    try:
        eval_node = lut.find_lut_tree_node(layer)
        eval_lut = eval_node.children[0].val
    except Exception:
        print("eval_table for layer: ", layer, " not found in LUT.")
    return eval_lut




