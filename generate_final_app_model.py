from converters.json_converters.json_app_config_parser import parse_app_conf, parse_json_dnns,\
    parse_json_mappings, partition_dnns_with_mapping
from models.dnn_model.transformation.external_ios_processor import external_ios_to_data_layers
import traceback
from util import print_stage
from models.app_model.MMSFinalAppModel import MMSFinalAppModel
from DSE.low_memory.mms.buf_building import get_mms_buffers


def build_final_app(app_config_path: str, best_chromosome_path: str, verbose=True):
    # parse config
    stage = "Parsing application configuration"
    conf = parse_app_conf(app_config_path)
    # print(conf)
    try:
        stage = "DNNs parsing"
        print_stage(stage, verbose)
        dnns = parse_json_dnns(conf["json_dnn_paths"])

        stage = "GA mappings parsing"
        print_stage(stage, verbose)
        dnn_mappings = parse_json_mappings(conf["json_mapping_paths"])

        # partition dnns according to the parsed mappings
        stage = "DNNs partitioning"
        print_stage(stage, verbose)
        partitions_per_dnn = partition_dnns_with_mapping(dnns, dnn_mappings)

        stage = "Representing external dnn data sources/consumers as (data) layers"
        print_stage(stage, verbose)
        # represent all external I/Os as explicit (data) layers: important for building
        # of inter-dnn buffers and external-source -> dnn buffers
        for partitions in partitions_per_dnn:
            for partition in partitions:
                external_ios_to_data_layers(partition)

        stage = "Creating final model"
        app_model = MMSFinalAppModel(conf["app_name"])
        app_model.dnns = dnns
        app_model.partitions_per_dnn = partitions_per_dnn

    except Exception as e:
        print("Final app generation error at stage " + stage)
        traceback.print_tb(e.__traceback__)


def tst():
    app_config_path = "./data/test/app_configs/single_dnn.json"
    best_chromosome_path = "./data/test/best_chromosome/single_dnn_app.json"
    build_final_app(app_config_path, best_chromosome_path)

tst()

