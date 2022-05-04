import json
from converters.json_converters.json_util import extract_or_default
from util import get_project_root
import os


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

            # set optional fields
            if "app_name" not in conf:
                conf_as_dict["app_name"] = "app"

            if "json_mapping_paths" not in conf:
                conf_as_dict["json_mapping_paths"] = [None for _ in range(len(conf["json_dnn_paths"]))]
            else:
                conf_as_dict["json_mapping_paths"] = conf["json_mapping_paths"]
                # replace empty strings (json representation of None) with None-values
                for elem_id in range(len(conf_as_dict["json_mapping_paths"])):
                    if conf_as_dict["json_mapping_paths"][elem_id] == "":
                        conf_as_dict["json_mapping_paths"][elem_id] = None

            if "output_file_path" not in conf:
                project_root_path = str(get_project_root())
                ga_output_path = str(os.path.join(project_root_path, "output", (conf["app_name"] + ".json")))
                conf_as_dict["output_file_path"] = ga_output_path

            return conf_as_dict


