from collections.abc import Sequence
from enum import Enum


class JSONNestedClassVisitor:
    """
    JSON visitor. Used to save custom classes as a .json file.
    Attributes:
        nested_class_object: object of a class, that may contain
        fields that are also objects of a (different) class
        filepath: path to target .json file
    """
    def __init__(self, nested_class_object, filepath):
        self._prefix = ""
        self.__prefix_len = 2
        self._root = nested_class_object
        self._filepath = filepath
        self._file = None

    def run(self):
        self._file = open(self._filepath, "w")
        try:
            self.visit_object(self._root)
        except Exception as e:
            print("Json visitor exception: ", str(e))
        self._file.close()

    def visit_object(self, obj):
        """
        Recursive object visitor
        :param obj: object to visit
        """
        if self.is_class_object(obj):
            if isinstance(obj, Enum):
                self.visit_enum(obj)
            else:
                self.visit_class_object(obj)
            return

        if self.is_dictionary(obj):
            self.__visit_dict(obj)
            return

        if self.is_collection(obj):
            self._visit_collection(obj)
            return

        else:
            self.visit_simple_object(obj)

    ###############################
    # type-specific visitors      #
    def visit_class_object(self, class_obj):
        # print("Visit class object: ", class_obj)
        self.__visit_dict(class_obj.__dict__)

    def visit_enum(self, obj):
        self._file.write("\"" + str(obj.name) + "\"")

    def __visit_dict(self, v_dict):
        self._file.write(self._prefix + "{\n")
        self.prefix_inc()
        item_id = 0
        max_items = len(v_dict.items())
        for item in v_dict.items():
            item_key = item[0]
            item_val = item[1]
            item_key = "\"" + item_key + "\": "
            self._file.write(self._prefix)
            self._file.write(item_key)
            self.visit_object(item_val)
            self.write_comma_sep_ln(item_id, max_items)
            item_id = item_id + 1
        self.prefix_dec()
        self._file.write("\n" + self._prefix + "}")

    def _visit_collection(self, v_list):
        self._file.write("[")
        self.prefix_inc()
        item_id = 0
        max_items = len(v_list)
        for elem in v_list:
            self.visit_object(elem)
            self.write_comma_sep(item_id, max_items)
            item_id = item_id + 1
        self.prefix_dec()
        self._file.write("]")

    def visit_simple_object(self, simple_obj):
        if isinstance(simple_obj, (str, bytes, bytearray)):
            self._file.write("\"" + simple_obj + "\"")
        elif isinstance(simple_obj, bool):
            if simple_obj is True:
                self._file.write("true")
            else:
                self._file.write("false")
        else:
            self._file.write(str(simple_obj))

    ###############################
    #           type checks      #

    def is_class_object(self, obj):
        try:
            if obj.__dict__:
                return True
        except AttributeError:
            return False

    def is_collection(self, obj):
        collection = isinstance(obj, Sequence)
        string = self.is_string(obj)
        return collection and not string

    def is_string(self, obj):
        return isinstance(obj, (str, bytes, bytearray))

    def is_dictionary(self, obj):
        return isinstance(obj, dict)

    ###############################
    #      utility funcs          #
    def write_comma_sep_ln(self, item_id, max_items):
        if item_id < max_items - 1:
            self._file.write(",\n")

    def write_comma_sep(self, item_id, max_items):
        if item_id < max_items - 1:
            self._file.write(",")

    def prefix_inc(self):
        prefix_addition = ""
        for i in range(self.__prefix_len):
            prefix_addition = prefix_addition + " "
        self._prefix = self._prefix + prefix_addition

    def prefix_dec(self):
        self._prefix = self._prefix[0: len(self._prefix) - self.__prefix_len]
