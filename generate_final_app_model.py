import argparse
import sys
import traceback
from os.path import dirname

"""
Console-interface script for the best chromosome selection performed after the GA-based search
"""


def main():
    parser = argparse.ArgumentParser(description='Train an onnx model with iterations')
    # required arguments
    parser.add_argument('-c', '--config', type=str, action='store',
                        help='path to .json application config', required=True)

    parser.add_argument('-b', '--best-chromosome', type=str, action='store',
                        help='path to best chromosome, generated using ./run_mms_selection.py '
                             'script and saved in .json format', required=True)

    parser.add_argument('-o', metavar='--output', type=str, action='store',
                        default="./output/best_chromosome/best_chromosome.json",
                        help='Path to the output .json file to save the best chromosome in.')

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
    from util import print_stage
    from DSE.low_memory.mms.final_model_building import build_final_app
    from converters.json_converters.json_app_config_parser import parse_app_conf, parse_json_dnns, \
        parse_json_mappings
    from converters.json_converters.mms_final_app_to_json import mms_app_to_json
    from fileworkers.json_fw import read_json

    try:
        # parse parameters
        conf_file = args.config
        output_file_path = args.o
        best_chromosome_path = args.b
        silent = args.silent
        verbose = not silent

        stage = "Reading best chromosome"
        print_stage(stage, verbose)
        json_best_chromosome = read_json(best_chromosome_path)
        dp_encoding = json_best_chromosome["dp_by_parts"]
        # print("dp_encoding:", dp_encoding)

        # parse config
        stage = "Parsing application configuration"
        print_stage(stage, verbose)
        conf = parse_app_conf(conf_file)

        stage = "DNNs parsing"
        print_stage(stage, verbose)
        dnns = parse_json_dnns(conf["json_dnn_paths"])

        stage = "GA mappings parsing"
        print_stage(stage, verbose)
        dnn_mappings = parse_json_mappings(conf["json_mapping_paths"])

        final_app_model = build_final_app(conf["app_name"], dnns, dnn_mappings, dp_encoding, verbose)
        # app_model.print_details()

        stage = "Saving final app model in JSON file (" + output_file_path + ")"
        print_stage(stage, verbose)
        mms_app_to_json(final_app_model, output_file_path)

    except Exception as e:
        print("MMS final app generation error")
        traceback.print_tb(e.__traceback__)


def tst():
    # single-dnn app (no pipeline)
    app_config_path = "./data/test/app_configs/single_dnn.json"
    best_chromosome_path = "./data/test/best_chromosome/single_dnn_app.json"
    output_file_path = "./output/single_dnn_final_app.json"


    """
    # single-dnn app (pipeline)
    app_config_path = "./data/test/app_configs/single_dnn_pipeline.json"
    best_chromosome_path = "./data/test/best_chromosome/single_dnn_app_pipeline.json"

    build_final_app(app_config_path, best_chromosome_path)

    # multi-dnn app (no pipeline)
    app_config_path = "./data/test/app_configs/multi_dnn.json"
    best_chromosome_path = "./data/test/best_chromosome/multi_dnn_app.json"

    build_final_app(app_config_path, best_chromosome_path)

    # multi-dnn app (pipeline)
    app_config_path = "./data/test/app_configs/multi_dnn_pipeline.json"
    best_chromosome_path = "./data/test/best_chromosome/multi_dnn_app_pipeline.json"

    build_final_app(app_config_path, best_chromosome_path)
    """


def get_cur_directory():
    this_dir = dirname(__file__)
    return this_dir


if __name__ == "__main__":
    main()

