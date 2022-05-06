from converters.json_converters.json_app_config_parser import parse_app_conf, parse_json_dnns,\
    parse_json_mappings, partition_dnns_with_mapping
from models.dnn_model.transformation.external_ios_processor import external_ios_to_data_layers
import traceback
from util import print_stage
from models.app_model.MMSDNNInferenceModel import MMSDNNInferenceModel
from DSE.low_memory.mms.buf_building import get_mms_buffers_and_schedule
from DSE.low_memory.mms.phases_derivation import dp_encoding_to_phases_num
from fileworkers.json_fw import read_json
from converters.data_buffers_converter import csdf_reuse_buf_to_generic_dnn_buf
from DSE.scheduling.dnn_scheduling import DNNScheduling
from simulation.csdf_simulation import simulate_execution_asap
from converters.dnn_to_csdf import dnn_to_csfd_one_to_one


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

        stage = "Reading best chromosome"
        print_stage(stage, verbose)
        json_best_chromosome = read_json(best_chromosome_path)
        dp_encoding = json_best_chromosome["dp_by_parts"]
        print("dp_encoding:", dp_encoding)

        stage = "Obtaining phases per dnn layer"
        print_stage(stage, verbose)
        phases = dp_encoding_to_phases_num(dp_encoding, dnns)
        print("phases per layer:", phases)

        stage = "Building DNN buffers and schedules through CNN-to-CSDF conversion and CSDF model analysis"
        print_stage(stage, verbose)
        csdf_buffers, csdf_schedule = get_mms_buffers_and_schedule(dnns,
                                                                   partitions_per_dnn,
                                                                   dp_encoding,
                                                                   generate_schedule=True,
                                                                   verbose=False)
        stage = "DNN  schedules (layers execution order)"
        print_stage(stage, verbose)
        if csdf_schedule is not None:
            for elem in csdf_schedule:
                print(elem)

        stage = "Creating DNN buffers description"
        print_stage(stage, verbose)
        generic_dnn_buffers = csdf_reuse_buf_to_generic_dnn_buf(csdf_buffers, dnns)
        # for buf in generic_dnn_buffers:
        #    buf.print_details()

        stage = "Creating final model"
        app_model = MMSDNNInferenceModel(conf["app_name"])
        app_model.dnn_names = [dnn.name for dnn in dnns]
        app_model.partitions_per_dnn = partitions_per_dnn
        app_model.phases_per_dnn_per_layer = phases
        # app_model.print_details()

    except Exception as e:
        print("Final app generation error at stage " + stage)
        traceback.print_tb(e.__traceback__)


def tst():
    # multi-dnn app (no pipeline)
    """
    app_config_path = "./data/test/app_configs/multi_dnn.json"
    best_chromosome_path = "./data/test/best_chromosome/multi_dnn_app.json"
    """

    # single-dnn app (no pipeline)
    app_config_path = "./data/test/app_configs/single_dnn.json"
    best_chromosome_path = "./data/test/best_chromosome/single_dnn_app.json"

    build_final_app(app_config_path, best_chromosome_path)

tst()

