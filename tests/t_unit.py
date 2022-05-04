import argparse
import os
import traceback
import sys


def main():
    """ Run parts of the pipeline for example CNN and example platform, defined in ../test_config.py"""
    # import current directory and it's subdirectories into system path for the current console
    # this would allow importing project modules without adding the project to the PYTHONPATH
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_dir)

    # general arguments
    parser = argparse.ArgumentParser(description='The script runs unit (per-step) tests for an example '
                                                 'CNN and an example platform, defined in ./test_config.py')
    parser.add_argument("-s", "--step", type=str, action='store', required=True,
                        help='Script (step) to test. Select from [ga_single_dnn]')

    parser.add_argument('--info-level', metavar='--info-level', type=int, action='store', default=1,
                        help='Info-level, i.e., amount of information to print out during the tests run. '
                             'If info-level == 0, no information is printed to console; '
                             'If info-level == 1, only tests information '
                             '(e.g., which steps of the tests were successful) is printed to console;'
                             'If info-level == 2, tests information (e.g., which steps of the tests were successful) '
                             'as well as script-specific verbose output is printed to the console')

    try:
        args = parser.parse_args()
        info_level = args.info_level
        step = args.step

        # import project modules
        from util import get_project_root
        from fileworkers.common_fw import clear_folder, create_or_overwrite_dir
        from tests.test_config import get_test_config

        unit_test_result = run_test_step(step, info_level)
        return unit_test_result

    except Exception as e:
        print(" Integration tests error: " + str(e))
        traceback.print_tb(e.__traceback__)
        return 1


def run_test_step(step: str, info_level):
    from tests.test_config import get_test_config
    config = get_test_config()
    if step == "ga_single_dnn":
        result = run_test_ga_single_dnn(config, info_level)
        return result

    raise Exception("Unknown tests step: " + step)


def run_test_ga_single_dnn(config: {}, info_level=1):
    """
    :param config: test app_config (see ../test_config.py)
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    app_config_path = str(os.path.join(config["app_configs_dir"], "single_dnn.json"))
    test_passed = run_test_case_ga(app_config_path, config, info_level,
                                   "Single-CNN application with no pipeline parallelism")
    return test_passed


def run_test_case_ga(app_config_path, test_config: {}, info_level, test_case_description: ""):
    """ Run test "run_mms_ga.py" script on test application app_config
    :param app_config_path: path to .json file with application configuration
    :param test_config: test app_config (see ../test_config.py)
        by subsequent scripts of the tool
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :param test_case_description: description of the test case (str). Printed into console before test is executed.
    :return: True if tests ran successfully and False otherwise
    """
    if info_level > 0:
        print("RUN GA-based search for test application (", test_case_description, ")")

    # import project modules
    from util import get_project_root
    from t_unit_common import run_script_and_check_output

    # test automatically fails if application app_config cannot be parsed
    app_config = try_parse_app_config(app_config_path, info_level)
    if app_config is None:
        test_passed = False
    else:
        if info_level > 0:
            print("  Run GA-based search")

        script_root = get_project_root()
        script_name = "run_mms_ga"
        abs_app_config_path = data_path_to_abs_data_path(app_config_path)
        input_param = {
            "-c": abs_app_config_path,
            "-t": str(test_config["cpu_threads"])
        }
        flags = []
        if info_level < 2:
            flags.append("--silent")

        output_file_abs_paths = [data_path_to_abs_data_path(app_config["output_file_path"])]

        test_passed = run_script_and_check_output(script_root,
                                                  script_name,
                                                  input_param,
                                                  flags,
                                                  output_file_abs_paths,
                                                  info_level)
    return test_passed


def data_path_to_abs_data_path(file_path):
    """
    Convert relative or absolute path to input data file
     into absolute path to input data file
    :param file_path: path to file
    :return: absolute path to file
    """
    # import project modules
    from util import get_project_root

    relative_path_prefixes = ["./../", "./"]

    for prefix in relative_path_prefixes:
        if file_path.startswith(prefix):
            file_path = file_path[len(prefix):]
            file_path = str(os.path.join(get_project_root(), file_path))
            return file_path

    return file_path


def try_parse_app_config(app_config_path, info_level):
    """
    Try to parse application config
    :param app_config_path: (absolute) path to the application config, saved as .json file
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: parsed application config if parsing was successful and None otherwise
    """
    from converters.json_converters.json_app_config_parser import parse_app_conf
    try:
        if info_level > 0:
            print("  Parse app config")
        conf = parse_app_conf(app_config_path)
        if conf is not None:
            if info_level > 0:
                print("  - SUCCESS")
            return conf
    except Exception as e:
        if info_level > 0:
            print("  App config parsing error: " + str(e))
            traceback.print_tb(e.__traceback__)

    if info_level > 0:
        print("  - FAILURE")
    return None


def run_test_ga_multi_dnn():
    from util import get_project_root
    from DSE.low_memory.mms.ga_based.multi_thread.mms_ga import run_ga_parallel_multi

    project_root_path = str(get_project_root())

    dnn1_path = project_root_path + "/data/json_dnn/CNN1.json"
    dnn2_path = project_root_path + "/data/json_dnn/CNN2.json"
    dnn_paths = [dnn1_path, dnn2_path]

    dnn1_mapping_path = None
    dnn2_mapping_path = project_root_path + "/data/pipeline_parallelism/CNN2.json"
    mapping_paths = [dnn1_mapping_path, dnn2_mapping_path]

    ga_conf_path = project_root_path + "/data/mms_ga_configs/testDNN_ga_conf.json"
    ga_output_path = project_root_path + "/output/cnn1_cnn2_no_pipeline_pareto.json"

    # script parameters
    parr_threads = 6

    # run script
    run_ga_parallel_multi(dnn_paths, mapping_paths, ga_conf_path, parr_threads, ga_output_path)


if __name__ == "__main__":
    main()

