import os
import subprocess
import sys


def run_script_and_check_output(script_root,
                                script_name,
                                input_param: {},
                                flags: [],
                                output_file_paths: [],
                                info_level):
    """
    Run script and check whether it was properly executed.
    If the script was properly executed it would generate a specific set of output files.
    :param script_root: folder, where script is located
    :param script_name: name of the script
    :param input_param: (dictionary) list of input parameters, where key=parameter name,
        value = parameter value
    :param flags: list of input flags
    :param output_file_paths: list of absolute paths to expected output files
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True, if the script was successfully executed and
        all the output files was generated, False otherwise
    """
    # run script
    script_successfully_executed = run_script(script_root, script_name, input_param, flags, info_level)
    if info_level > 0:
        print("  - script successfully executed:", script_successfully_executed)

    # check the output files
    output_files_generated = files_exist(output_file_paths)
    if info_level > 0:
        print("  - script output files are successfully generated:", output_files_generated)

    test_passed = True if script_successfully_executed and output_files_generated else False
    if info_level > 0:
        if test_passed:
            print("TEST PASSED")
        else:
            print("TEST FAILED")
    return test_passed


def run_script(script_root, script_name, input_param: {}, flags: [], info_level):
    """
    Run script
    :param script_root: folder, where script is located
    :param script_name: name of the script
    :param input_param: (dictionary) list of input parameters, where key=parameter name,
        value = parameter value
    :param flags: list of input flags
    :param info_level: amount of information to print out during the tests run.
        If info-level == 0, no information is printed to console.
        If info-level == 1, only tests information (e.g., which steps of the tests were successful)
        is printed to console.
        If info-level == 2, tests information (e.g., which steps of the tests were successful)
        as well as script-specific verbose output is printed to the console
    :return: True, if the list of output files was generated, False otherwise
    """
    script_path = str(os.path.join(script_root, (script_name + ".py")))
    # define script call. The first parameter is the path to (current) python interpreter.
    # The second parameter is the (absolute) path to executable script
    script_call = [sys.executable, script_path]

    # add parameters
    for item in input_param.items():
        param, val = item
        script_call.append(param)
        script_call.append(val)

    # add flags
    for flag in flags:
        script_call.append(flag)

    # print("call param", script_call)
    result = subprocess.run(script_call, capture_output=True, text=True)
    success = True if result.returncode == 0 else False
    # print("success:", success)

    # print stdout and stderr returned by script
    if info_level > 0:
        print("  - script stdout:", result.stdout)
        print("  - script stderr:", result.stderr)

    return success


def files_exist(files_abs_paths):
    """
    Check if files exist
    :param files_abs_paths abs paths to files
    :return: True if all the files exist, False otherwise
    """
    for file_path in files_abs_paths:
        if not os.path.exists(file_path):
            return False
    return True

