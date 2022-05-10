import json
from converters.json_converters.json_util import extract_or_default
from util import get_project_root
import os
from converters.json_converters.json_to_dnn import parse_json_dnn
from converters.json_converters.json_mapping_parser import parse_mapping
from DSE.partitioning.after_mapping.partition_dnn_with_mapping import partition_dnn_with_mapping
from models.dnn_model.dnn import DNN


def parse_app_conf(path):
    """ Parse max-memory-save (MMS) GA application config """
    with open(path, 'r') as file:
        if file is None:
            raise FileNotFoundError
        else:
            conf_as_dict = {}
            conf = json.load(file)
            required_fields = ["json_dnn_paths", "json_ga_conf_path"]
            optional_fields = ["app_name", "json_mapping_paths", "output_file_path"]

            for required_field in required_fields:
                if required_field not in conf:
                    raise Exception("Required field " + required_field + " not found in application config")
                else:
                    conf_as_dict[required_field] = conf[required_field]

            # make paths to dnns and to ga config absolute
            conf_as_dict["json_dnn_paths"] = [path_from_project_root_to_abs_path(file_path) for
                                              file_path in conf_as_dict["json_dnn_paths"]]
            conf_as_dict["json_ga_conf_path"] = path_from_project_root_to_abs_path(conf_as_dict["json_ga_conf_path"])

            # set optional fields
            conf_as_dict["app_name"] = extract_or_default(conf, "app_name", "app")
            conf_as_dict["json_mapping_paths"] = get_json_mapping_paths(conf)
            conf_as_dict["output_file_path"] = get_output_file_path(conf)

            return conf_as_dict


def get_json_mapping_paths(json_app_config):
    # unspecified mapping paths (no pipeline mapping is used for the application)
    if "json_mapping_paths" not in json_app_config:
        json_mapping_paths = [None for _ in range(len(json_app_config["json_dnn_paths"]))]
        return json_mapping_paths

    # specified mapping paths
    json_mapping_paths = json_app_config["json_mapping_paths"]

    # replace relative paths with abs paths
    json_mapping_paths_abs = [path_from_project_root_to_abs_path(file_path) for file_path in json_mapping_paths]

    # replace empty strings (json representation of None) with None-values
    json_mapping_paths_abs_formatted = [string_to_none_if_empty(file_path) for file_path in json_mapping_paths_abs]
    return json_mapping_paths_abs_formatted


def get_output_file_path(json_app_config):
    default_output_file_path = str(os.path.join(get_project_root(), "output",
                                                (json_app_config["app_name"] + "_chromosomes.json")))
    output_file_path = extract_or_default(json_app_config, "output_file_path", default_output_file_path)
    output_file_path = path_from_project_root_to_abs_path(output_file_path)
    return output_file_path


def path_from_project_root_to_abs_path(file_path: str) -> str:
    """
    Convert relative or absolute path to input data file
     into absolute path to input data file
    :param file_path: path to file
    :return: absolute path to file
    """

    # import project modules
    from util import get_project_root
    project_root = get_project_root()

    relative_path_prefixes = ["./", "MMSROOT/"]

    for prefix in relative_path_prefixes:
        if file_path.startswith(prefix):
            file_path = file_path[len(prefix):]
            file_path = str(os.path.join(project_root, file_path))
            return file_path

    return file_path


def string_to_none_if_empty(input_str):
    """
    :param input_str: input string
    :return: None if input strings is empty (="") and input string otherwise
    """
    if input_str == "":
        return None
    else:
        return input_str


def parse_json_dnns(json_dnn_paths: []):
    """
    Parse DNNs, presented in JSON format
    :param json_dnn_paths: list, where every element is a path(str) to a DNN
        saved in .json file
    :return: list of DNNs, where every DNN is represented as an object
        of DNN class (see models/dnn_model/dnn.py)
    """
    dnns = []
    for json_dnn_path in json_dnn_paths:
        dnn = parse_json_dnn(json_dnn_path)
        dnns.append(dnn)
    return dnns


def parse_json_mappings(json_mapping_paths):
    """
    Parse dnn mappings
    :param json_mapping_paths: list, where every element is a path to dnn mapping,
        saved in JSON format
    :return: list of json mappings, where every mapping is a two-dimensional array of integers
    """
    dnn_mappings = []
    for json_mapping_path in json_mapping_paths:
        if json_mapping_path is None:
            dnn_mappings.append(None)
        else:
            mapping = parse_mapping(json_mapping_path)
            dnn_mappings.append(mapping)
            # print("mapping parsed: ", mapping)
    return dnn_mappings


def partition_dnns_with_mapping(dnns: [DNN], dnn_mappings):
    """
    Partition DNNs with pipeline mapping
    :param dnns: list of DNNs, where every DNN is an object of DNN class (see models/dnn_model/dnn.py)
    :param dnn_mappings: list of json mappings, where every mapping is a two-dimensional array of integers
    :return: list of partitioned DNNs, where every partitioned DNN is a list of DNN partitions (sub-networks)
    """
    partitions_per_dnn = []
    for dnn_id in range(len(dnns)):
        dnn = dnns[dnn_id]
        mapping = dnn_mappings[dnn_id]
        if mapping is None:
            # no pipeline mapping
            dnn_partitions = [dnn]
        else:
            dnn_partitions, inter_dnn_connections = partition_dnn_with_mapping(dnn, mapping)
        partitions_per_dnn.append(dnn_partitions)
    return partitions_per_dnn




