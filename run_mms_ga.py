import argparse
from os.path import dirname
import sys
import traceback

"""
Console-interface script for GA-based search
"""


def main():
    parser = argparse.ArgumentParser(description='Train an onnx model with iterations')
    # required arguments
    parser.add_argument('-c', '--config', type=str, action='store',
                        help='path to .json application config', required=True)
    parser.add_argument('-t', '--threads', type=int, action='store',
                        help='number of parallel CPU threads', required=True)
    # general flags
    parser.add_argument("--silent", help="do not provide print-out for the script steps",
                        action="store_true", default=False)

    # parse arguments
    args = parser.parse_args()

    # Determine current directory and add path to this
    # directory to syspath to use other .python modules
    this_dir = get_cur_directory()
    sys.path.append(this_dir)

    # import sub-modules
    from DSE.low_memory.mms.ga_based.multi_thread.mms_ga import run_ga_parallel_multi
    from converters.json_converters.json_app_config_parser import parse_app_conf
    from util import print_stage

    try:
        # parse config
        conf_file = args.config
        parr_threads = args.threads
        silent = args.silent
        verbose = not silent

        stage = "Parsing application configuration"
        print_stage(stage, verbose)
        conf = parse_app_conf(conf_file)
        if verbose:
            print("script executed with config: ")
            for item in conf.items():
                print(item)

        # run script
        run_ga_parallel_multi(conf["json_dnn_paths"],
                              conf["json_mapping_paths"],
                              conf["json_ga_conf_path"],
                              parr_threads,
                              conf["output_file_path"],
                              verbose)

    except Exception as e:
        print("GA-based search error: " + str(e))
        traceback.print_tb(e.__traceback__)


def get_cur_directory():
    this_dir = dirname(__file__)
    return this_dir


if __name__ == "__main__":
    main()

