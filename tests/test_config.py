import os
from util import get_project_root


def get_test_config():
    # path to input files
    input_files_folder_abs = os.path.join(get_project_root(), "data")

    # path to input cnns ( each cnn is a json file)
    cnns_dir = str(os.path.join(input_files_folder_abs, "json_dnn"))

    # path to input app configs ( each app config is a json file)
    app_configs_dir = str(os.path.join(input_files_folder_abs, "test", "app_configs"))

    # path to input GA configs ( each app config is a json file)
    ga_configs_dir = str(os.path.join(input_files_folder_abs, "mms_ga_configs"))

    # path to intermediate files, produced by steps of the tool during the tests
    intermediate_files_folder_abs = os.path.join(get_project_root(), "output", "test")

    test_config = {
        "input_files_folder_abs": input_files_folder_abs,
        "app_configs_dir": app_configs_dir,
        "intermediate_files_folder_abs": intermediate_files_folder_abs,
        "cpu_threads": 6
    }
    return test_config

