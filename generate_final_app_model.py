from converters.json_converters.json_to_dnn import parse_json_dnn
from converters.json_converters.json_mms_ga_conf_parser import parse_mms_ga_conf
from converters.json_converters.json_mapping_parser import parse_mapping
from converters.json_converters.json_app_config_parser import parse_app_conf
from converters.json_converters.mms_chromosomes_to_json import mms_chromosomes_to_json
from dnn_partitioning.after_mapping.partition_dnn_with_mapping import partition_dnn_with_mapping
from models.dnn_model.transformation.external_ios_processor import external_ios_to_data_layers
import traceback
from util import print_to_stderr
from DSE.low_memory.mms.ga_based.multi_thread.MMSgaParallelMultiPipeline import MMSgaParallelMultiPipeline


def build_final_app(app_config_path: str, best_chromosome_path: str):
    # parse config
    stage = "Parsing application configuration"
    conf = parse_app_conf(app_config_path)

    stage = "DNNs parsing"
    try:
        dnns = []
        for json_dnn_path in conf["json_dnn_paths"]:
            dnn = parse_json_dnn(json_dnn_path)
            dnns.append(dnn)
        # dnn.print_details()

        stage = "GA mappings parsing"
        dnn_mappings = []
        for json_mapping_path in conf["json_mapping_paths"]:
            if json_mapping_path is None:
                dnn_mappings.append(None)
            else:
                mapping = parse_mapping(json_mapping_path)
                dnn_mappings.append(mapping)
                # print("mapping parsed: ", mapping)

        # partition dnns according to the parsed mappings
        stage = "DNNs partitioning"
        partitions_per_dnn = []
        for dnn_id in range(len(dnns)):
            dnn = dnns[dnn_id]
            mapping = dnn_mappings[dnn_id]
            if mapping is None:
                # no pipeline mapping
                dnn_partitions = [dnn]
            else:
                dnn_partitions, inter_dnn_connections = partition_dnn_with_mapping(dnn, mapping)
            partitions_per_dnn.append(dnn_partitions)

        stage = "Representing external dnn data sources/consumers as (data) layers"
        # represent all external I/Os as explicit (data) layers: important for building
        # of inter-dnn buffers and external-source -> dnn buffers
        for partitions in partitions_per_dnn:
            for partition in partitions:
                external_ios_to_data_layers(partition)

    except Exception as e:
        print("GA-based search error: " + str(e))
        traceback.print_tb(e.__traceback__)