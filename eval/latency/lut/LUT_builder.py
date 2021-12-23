"""
This module is responsible for building LUT with distinctive
DNN layers for LUT-based power/perf DNN evaluation.
For memory/traverse time efficiency the DNN LUT is represented
as a treeMap of fixed height. The LUT can be printed as table, if needed.
"""
from dnn_model.dnn import DNN, Layer
from converters.parsers.json_benchmark_parser import parse_jetson_bm_as_annotated_dnn


def build_lut_tree_from_jetson_benchmark(bm_path):
    """
    Build LUT tree from benchmark and annotate it with time
    :param bm_path: benchmark path
    :return: LUT tree, annotated with time
    """
    annotated_dnn = parse_jetson_bm_as_annotated_dnn(bm_path)
    lut = build_lut_tree([annotated_dnn])
    # lut.print_as_table()
    return lut


def build_lut_tree(dnns: [DNN]):
    """
    Get LUT tree for a DNN
    """
    lut_tree = LUTTree()

    for dnn in dnns:
        for layer in dnn.get_layers():
            lut_tree.append_layer(layer)

    return lut_tree


def check_lut_tree(dnns: [DNN], lut_tree, verbose) -> bool:
    """
    Check, that LUT tree is consistent. The LUt tree is consistent, if
    it has entry for every DNN layer, and is inconsistent otherwise.
    """
    consistent = True
    for dnn in dnns:
        for layer in dnn.get_layers():
            node = lut_tree.find_lut_tree_node(layer, verbose)
            if node is None:
                consistent = False
    if verbose:
        print("tree is consistent: " + str(consistent))

    return consistent


class LUTTree:
    """
    Class, representing LUT tree
    """
    def __init__(self):
        self.root = LUTTreeNode("root", "root", None)

    def append_layer(self, layer: Layer):
        self.append_lut_tree_node(layer)

    def append_lut_tree_node(self, layer):
        """
        Append node in LUTTree
        """
        pto = param_traverse_order()  # parameter names in traverse order
        lpto = get_layer_param_in_traverse_order(
            layer)  # dictionary with k-v : param name - param value for a dnn layer
        cur_node = self.root

        for name in pto:
            val = lpto[name]
            # go to next level in LUTTree. If there is no branch, corresponding to the layer, create the branch
            cur_node = cur_node.find_or_append_child_node(name, val)

    def find_lut_tree_node(self, layer, verbose=False):
        """
        Find node in LUTTree, that corresponds to a dnn layer
        @ layer : dnn layer
        @:return node of tree (if found) and None otherwise
        """
        pto = param_traverse_order()  # parameter names in traverse order
        # dictionary with k-v : param name - param value for a dnn layer
        lpto = get_layer_param_in_traverse_order(layer)
        cur_node = self.root

        for name in pto:
            val = lpto[name]
            if name == "time":
                return cur_node.children[0]

            # the node is not found after traversing all the tree
            if cur_node is None:
                if verbose:
                    print("node", name, ":", val, "for layer", layer, "not found in the tree. None is returned."
                                                                      " Search stopped at param", name)
                return None

            # go to next level in LUTTree.
            cur_node = cur_node.find_child_node(name, val)
        return cur_node

    """
    Get elements of LUTTree as a benchmark Table
    """
    def get_as_table(self):
        entries = []
        self.__build_entry_list(self.root, entries)
        return entries

    def __build_entry_list(self, root, entries: []):
        """
        Print list of all the leafs (recoursive)
        """
        #not a leaf
        if root.children:
            for child in root.children:
                self.__build_entry_list(child, entries)

        #leaf
        else:
            path_node_to_root = self._get_path_to_root(root)
            path_root_to_node = self.__sort_path_to_root_top_down(path_node_to_root)
            entries.append(path_root_to_node)

    def _get_path_to_root(self, node):
        """
        Get path from node to root
        """
        cur_node = node
        result = {}
        while cur_node.parent is not None:
            result[cur_node.name] = cur_node.val
            cur_node = cur_node.parent

        return result

    def __sort_path_to_root_top_down(self, path_to_root: {}):
        pto = param_traverse_order()
        sorted_path = sort_lut_record(path_to_root, pto)
        return sorted_path

    """
    Get benchmark table size
    @:return benchmark table size
    """
    def get_table_size(self):
        entry_list = self.get_as_table()
        return len(entry_list)

    """
    Print elements of LUTTree(Traverse LUTTree with DFS)
    """
    def print_tree(self, root="root", prefix=""):
        if root == "root":
            root = self.root

        # print root node
        print(prefix, root.name, ":", root.val)
        prefix = prefix + "  "
        for child in root.children:
            self.print_tree(child, prefix)

    """
    Print elements of LUTTree as a benchmark Table
    """
    def print_as_table(self):
        entry_list = self.get_as_table()
        for entry in entry_list:
            print(entry)


class LUTTreeNode:
    """
    Class, representing node of the building blocks tree
    Every node has one root and zero or more children. The list of children is
    sorted from min to max value (if applicable)
    """
    def __init__(self, name, value, parent):
        self.name = name
        self.val = value
        self.parent = parent
        self.children = []
        #time evaluation: only available for leaf nodes
        self.time_eval = 0

    def find_or_append_child_node(self, name, value):
        """
        Get child node, that contain name-value pair, if it is in child nodes list
        If child nodes list does not contain such a node, append such node to the children list'
        :return node, that contain name-value pair and is in the children nodes list
        """
        n = self.find_child_node(name, value)
        if n is None:
            n = self.append_child_node(name, value)
        return n

    """
    Append new child node to the child nodes list
    @:return appended child node
    """
    def append_child_node(self, name, value):
        n = LUTTreeNode(name, value, self)
        self.children.append(n)
        return n

    """
    Get child node, that contain name-value pair or return None
    """
    def find_child_node(self, name, value):
        for n in self.children:
            if n.name == name and n.val == value:
                return n
        return None


def sort_lut_record(record, ordered_keys):
    sorted_record = {}
    for key in ordered_keys:
        sorted_record[key] = record[key]
    return sorted_record


def param_traverse_order():
    """
    Traverse order of parameters in the building blocks tree
    """
    pto = ["op", "fs", "stride", "ifm", "ofm", "iw", "wpad", "hpad", "time"]
    return pto


def get_layer_param_in_traverse_order(layer: Layer):
    """
    Get dnn layer parameters in LUTTree traverse order
    :param layer dnn layer, defined as object of class Layer
    """
    lpto = {}
    lpto["op"] = layer.op
    lpto["fs"] = layer.fs
    lpto["stride"] = layer.stride
    lpto["ifm"] = layer.ifm
    lpto["ofm"] = layer.ofm
    lpto["iw"] = layer.iw
    lpto["wpad"] = int(layer.pads[0])
    lpto["hpad"] = int(layer.pads[1])
    lpto["time"] = layer.time_eval
    return lpto



