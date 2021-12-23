from models.dnn_model.dnn import DNN, Layer, Connection, ExternalInputConnection, ExternalOutputConnection
from converters.json_converters.JSONNestedClassVisitor import JSONNestedClassVisitor


def dnn_to_json(dnn: DNN, filepath: str):
    """
    Convert (analytical) DNN model into a JSON File
    :param dnn: (analytical) DNN model
    :param filepath: path to target .json file
    :return: JSON string, encoding the analytical DNN model
    """
    visitor = DNNJSONNestedClassVisitor(dnn, filepath)
    visitor.run()


def get_layer_ref(layer: Layer):
    """
    Get JSON reference to a DNN layer.
    :param layer: DNN layer
    :return: reference to the layer
    """
    return layer.id


class DNNJSONNestedClassVisitor (JSONNestedClassVisitor):
    """
    JSON visitor. Used to save custom classes as a .json file.
    Attributes:
        dnn: object of DNN class
        filepath: path to target .json file
    """
    def __init__(self, dnn: DNN, filepath):
        super(DNNJSONNestedClassVisitor, self).__init__(dnn, filepath)
        # represent object of a dnn class as dictionary of fields
        self.dnn_as_dict = self._root.__dict__
        # number of fields in the dnn class object represented as a dictionary
        self.dnn_fields_num = len(self.dnn_as_dict.items())
        self.cur_dnn_field_id = 0

    def visit_object(self, obj):
        """
        Recursive object visitor
        :param obj: object to visit
        """
        if isinstance(obj, Connection):
            self._visit_connection(obj)
            return

        if isinstance(obj, ExternalInputConnection):
            self._visit_external_io_connection(obj)
            return

        if isinstance(obj, ExternalOutputConnection):
            self._visit_external_io_connection(obj)
            return

        super().visit_object(obj)

    def _visit_connection(self, connection_obj):
        v_dict = connection_obj.__dict__
        # only save src and dst layers
        v_dict_trimmed = {"src": v_dict["src"],
                          "dst": v_dict["dst"]}

        self._file.write(self._prefix + "{\n")
        self.prefix_inc()
        item_id = 0
        max_items = len(v_dict_trimmed.items())
        for item in v_dict_trimmed.items():
            item_key = item[0]
            item_val = item[1]
            item_key = "\"" + item_key + "\": "
            self._file.write(self._prefix)
            self._file.write(item_key)

            # replace layer with layer reference
            if isinstance(item_val, Layer):
                item_val = get_layer_ref(item_val)

            self.visit_object(item_val)
            self.write_comma_sep_ln(item_id, max_items)
            item_id = item_id + 1
        self.prefix_dec()
        self._file.write("\n" + self._prefix + "}")

    def _visit_external_io_connection(self, connection_obj):
        v_dict = connection_obj.__dict__
        self._file.write(self._prefix + "{\n")
        self.prefix_inc()
        item_id = 0
        max_items = len(v_dict.items())
        for item in v_dict.items():
            item_key, item_val = item
            item_key = "\"" + item_key + "\": "
            self._file.write(self._prefix)
            self._file.write(item_key)

            # replace layer with layer reference
            if item[0] == "dnn_layer":
                item_val = get_layer_ref(item_val)

            self.visit_object(item_val)
            self.write_comma_sep_ln(item_id, max_items)
            item_id = item_id + 1
        self.prefix_dec()
        self._file.write("\n" + self._prefix + "}")
