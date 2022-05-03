import os.path


def run_test_ga():
    pass


def run_test_ga_single_dnn():
    # import project modules
    from DSE.low_memory.mms.ga_based.multi_thread.mms_ga import run_ga
    from tests.test_config import get_test_config

    test_config = get_test_config()

    test_json_dnn_path = str(os.path.join(test_config["cnns_dir"], "CNN1.json"))
    test_ga_conf_path = str(os.path.join(test_config["ga_configs_dir"], "testDNN_ga_conf.json"))
    test_ga_output_path = str(os.path.join(test_config["intermediate_files_folder_abs"], "testDNN_pareto.json"))
    parr_threads = test_config["cpu_threads"]

    run_ga(test_json_dnn_path, test_ga_conf_path, parr_threads, test_ga_output_path)


def run_test_ga_multi_dnn():
    from util import get_project_root
    from DSE.low_memory.mms.ga_based.multi_thread.mms_ga import run_ga_parallel_multi

    project_root_path = str(get_project_root())

    dnn1_path = project_root_path + "/data/json_dnn/CNN1.json"
    dnn2_path = project_root_path + "/data/json_dnn/CNN2.json"
    dnn_paths = [dnn1_path, dnn2_path]

    dnn1_mapping_path = None
    dnn2_mapping_path = project_root_path + "/data/pipeline_mapping/CNN2.json"
    mapping_paths = [dnn1_mapping_path, dnn2_mapping_path]

    ga_conf_path = project_root_path + "/data/mms_ga_configs/testDNN_ga_conf.json"
    ga_output_path = project_root_path + "/output/cnn1_cnn2_no_pipeline_pareto.json"

    # script parameters
    parr_threads = 6

    # run script
    run_ga_parallel_multi(dnn_paths, mapping_paths, ga_conf_path, parr_threads, ga_output_path)


def test_conf_file_parsing():
    from converters.json_converters.json_app_config_parser import parse_app_conf
    from util import get_project_root

    project_root_path = str(get_project_root())
    conf_file = project_root_path + "/data/app_configs/test_app_conf.json"

    conf = parse_app_conf(conf_file)
    print("script executed with config: ")
    for item in conf.items():
        print(item)

# run_test()
# run_test_multi()
# run_mobilenet_v2()
# test_conf_file_parsing()