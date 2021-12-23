import argparse
from os.path import dirname, join
import os
import sys

"""
Console-interface script for run/debug DNN training
"""

# example use on local machine
# python3 main.py -c ./config/local_config.json


def main():
    parser = argparse.ArgumentParser(description='Train an onnx model with iterations')
    # required arguments
    parser.add_argument('-c', '--config', type=str, action='store', help='path to .json config', required=True)
    parser.add_argument('-t', '--threads', type=int, action='store',
                        help='number of parallel CPU threads', required=True)

    # parse arguments
    args = parser.parse_args()

    # Determine current directory and add path to this
    # directory to syspath to use other .python modules
    this_dir = get_cur_directory()
    sys.path.append(this_dir)

    # import sub-modules
    from run_ga import run_ga_parallel_multi
    from converters.json_converters.json_app_config_parser import parse_app_conf

    # parse config
    conf_file = args.config
    parr_threads = args.threads
    conf = parse_app_conf(conf_file)

    print("script executed with config: ")
    for item in conf.items():
        print(item)

    # run script
    run_ga_parallel_multi(conf["json_dnn_paths"],
                          conf["json_mapping_paths"],
                          conf["json_ga_conf_path"],
                          parr_threads,
                          conf["output_file_path"])


def get_cur_directory():
    this_dir = dirname(__file__)
    return this_dir


if __name__ == "__main__":
    main()

