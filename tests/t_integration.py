import argparse
import os
import traceback
import sys


def main():
    """ Run the whole pipeline for example CNN-based applications defined in ./data/test/app_configs"""
    # import current directory and it's subdirectories into system path for the current console
    # this would allow importing project modules without adding the project to the PYTHONPATH
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(root_dir)

    # general arguments
    parser = argparse.ArgumentParser(description='The script runs an integration (end-to-end) tests for for an example '
                                                 'CNN and an example platform, defined in ./test_config.py')

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
        integration_test_result = run_integration_test(info_level)
        return integration_test_result

    except Exception as e:
        print(" Integration tests error: " + str(e))
        traceback.print_tb(e.__traceback__)
        return 1


def run_integration_test(info_level):
    """
    Run integration tests
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True if tests ran successfully and False otherwise
    """
    # import project modules
    from fileworkers.common_fw import clear_folder, create_or_overwrite_dir
    from tests.test_config import get_test_config
    from tests.t_unit import run_test_step

    if info_level > 0:
        print("RUN INTEGRATION TEST")

    # get test config
    config = get_test_config()

    # cleanup test output files
    test_output_files_folder = config["intermediate_files_folder_abs"]
    if info_level > 0:
        print("Clean target files folder", test_output_files_folder)
    clear_folder(test_output_files_folder)
    create_or_overwrite_dir(test_output_files_folder)

    # steps (scripts) in execution order
    steps = ["ga_single_dnn", "ga_single_dnn_pipeline",
             "ga_multi_dnn", "ga_multi_dnn_pipeline",
             "selection_single_dnn", "selection_single_dnn_pipeline",
             "selection_multi_dnn", "selection_multi_dnn_pipeline",
             "final_app_single_dnn", "final_app_single_dnn_pipeline",
             "final_app_multi_dnn", "final_app_multi_dnn_pipeline"]
    for step in steps:
        step_executed = run_test_step(step, info_level)
        if step_executed is False:
            if info_level > 0:
                print("INTEGRATION TEST FAILED")
                return 1
    print("INTEGRATION TEST PASSED")
    return 0


if __name__ == "__main__":
    main()

