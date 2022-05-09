from converters.json_converters.json_app_config_parser import parse_app_conf, parse_json_dnns,\
    parse_json_mappings, partition_dnns_with_mapping
from models.dnn_model.transformation.external_ios_processor import external_ios_to_data_layers
import traceback
from util import print_stage
from models.app_model.MMSDNNInferenceModel import MMSDNNInferenceModel
from DSE.low_memory.mms.buf_building import get_mms_buffers_and_schedule
from DSE.low_memory.mms.phases_derivation import dp_encoding_to_phases_num
from converters.data_buffers_converter import csdf_reuse_buf_to_generic_dnn_buf
from models.dnn_model.dnn import DNN


def build_final_app(app_name: str,
                    dnns: [DNN],
                    dnn_pipeline_mappings: [],
                    dp_encoding: [bool],
                    verbose=True):
    # parse config
    stage = "Parsing application configuration"

    # print(conf)
    try:
        # partition dnns according to the parsed mappings
        stage = "DNNs partitioning"
        print_stage(stage, verbose)
        partitions_per_dnn = partition_dnns_with_mapping(dnns, dnn_pipeline_mappings)

        stage = "Representing external dnn data sources/consumers as (data) layers"
        print_stage(stage, verbose)
        # represent all external I/Os as explicit (data) layers: important for building
        # of inter-dnn buffers and external-source -> dnn buffers
        for partitions in partitions_per_dnn:
            for partition in partitions:
                external_ios_to_data_layers(partition)

        stage = "Obtaining phases per dnn layer"
        print_stage(stage, verbose)
        phases = dp_encoding_to_phases_num(dp_encoding, dnns)
        # print("phases per layer:", phases)

        stage = "Building DNN buffers and per-partition schedules through CNN-to-CSDF " \
                "conversion and CSDF model analysis"
        print_stage(stage, verbose)
        csdf_buffers, app_schedule = get_mms_buffers_and_schedule(dnns,
                                                                  partitions_per_dnn,
                                                                  dp_encoding,
                                                                  generate_schedule=True,
                                                                  verbose=False)

        stage = "Creating DNN buffers description"
        print_stage(stage, verbose)
        generic_dnn_buffers = csdf_reuse_buf_to_generic_dnn_buf(csdf_buffers, dnns)
        # for buf in generic_dnn_buffers:
        #    buf.print_details()

        stage = "Creating final model"
        print_stage(stage, verbose)
        app_model = MMSDNNInferenceModel(app_name,
                                         [dnn.name for dnn in dnns],
                                         partitions_per_dnn,
                                         app_schedule,
                                         generic_dnn_buffers,
                                         phases
                                         )
        return app_model

    except Exception as e:
        print("MMS final app generation error at stage " + stage)
        traceback.print_tb(e.__traceback__)
        return None
