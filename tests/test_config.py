import os
from util import get_project_root


def get_test_config():
    # path to input files
    input_files_folder_abs = os.path.join(get_project_root(), "data")

    # path to input app configs ( each app config is a json file)
    app_configs_dir = str(os.path.join(input_files_folder_abs, "test", "app_configs"))

    # path to best chromosomes (each saved in a .json file)
    best_chromosomes_dir = str(os.path.join(input_files_folder_abs, "test", "best_chromosome"))

    # path to pareto-points encoded as chromosomes and saved in a .json file
    pareto_dir = str(os.path.join(input_files_folder_abs, "test", "pareto"))

    # path to intermediate files, produced by steps of the tool during the tests
    intermediate_files_folder_abs = os.path.join(get_project_root(), "output", "test")

    test_config = {
        "input_files_folder_abs": input_files_folder_abs,
        "app_configs_dir": app_configs_dir,
        "best_chromosomes_dir": best_chromosomes_dir,
        "intermediate_files_folder_abs": intermediate_files_folder_abs,
        "pareto_dir": pareto_dir,
        "cpu_threads": 6
    }
    return test_config

