def run_ga_parallel_multi(json_dnn_paths,
                          json_mapping_paths,
                          json_ga_conf_path,
                          parr_threads,
                          output_file_path,
                          verbose=True):
    """
    Run max memory save (mms) GA with support for multi-dnn and pipelined applications
    :param json_dnn_paths: list of paths to DNN models saved in .json format
    :param json_mapping_paths: list of paths to DNN model pipeline mapping saved in .json format
    :param json_ga_conf_path: path to json GA config for MMS GA
    :param parr_threads: parallel CPU threads to run GA on
    :param output_file_path: path to output json file, where the
     pareto-front of MMS-GA chromosomes, delivered by MMS-GA will be saved
    :param verbose: print GA execution details into console
    """
    from converters.json_converters.json_to_dnn import parse_json_dnn
    from converters.json_converters.json_mms_ga_conf_parser import parse_mms_ga_conf
    from converters.json_converters.json_mapping_parser import parse_mapping
    from converters.json_converters.mms_chromosomes_to_json import mms_chromosomes_to_json
    from dnn_partitioning.after_mapping.partition_dnn_with_mapping import partition_dnn_with_mapping
    from models.dnn_model.transformation.external_ios_processor import external_ios_to_data_layers
    import traceback
    from util import print_to_stderr
    from DSE.low_memory.mms.ga_based.multi_thread.MMSgaParallelMultiPipeline import MMSgaParallelMultiPipeline

    stage = "DNNs parsing"
    try:
        dnns = []
        for json_dnn_path in json_dnn_paths:
            dnn = parse_json_dnn(json_dnn_path)
            dnns.append(dnn)
        # dnn.print_details()

        stage = "GA mappings parsing"
        dnn_mappings = []
        for json_mapping_path in json_mapping_paths:
            if json_mapping_path is None:
                dnn_mappings.append(None)
            else:
                mapping = parse_mapping(json_mapping_path)
                dnn_mappings.append(mapping)
                # print("mapping parsed: ", mapping)

        stage = "GA config parsing"
        conf = parse_mms_ga_conf(json_ga_conf_path)
        # print(conf)

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

        stage = "GA setup"
        ga = MMSgaParallelMultiPipeline(partitions_per_dnn,
                                        conf["epochs"],
                                        conf["population_start_size"],
                                        conf["selection_percent"],
                                        conf["mutation_probability"],
                                        conf["mutation_percent"],
                                        conf["max_no_improvement_epochs"],
                                        conf["dp_by_parts_init_probability"],
                                        conf["data_token_size"],
                                        parr_threads,
                                        verbose)

        stage = "GA initialization with first population"
        ga.init_with_random_population()

        # for chromosome in ga.population:
        #    chromosome.print_long()

        stage = "GA execution"
        pareto_front = ga.run()
        if verbose:
            print("GA returned pareto front of", len(pareto_front), "elements, with min-buffers chromosome:")

        # best (in terms of buffers sizes) chromosome
        best_chromosome = pareto_front[0]

        if verbose:
            print(best_chromosome.dp_by_parts)

        stage = "Output JSON file generation"
        mms_chromosomes_to_json(pareto_front, output_file_path)

    except Exception:
        print_to_stderr("Exception at stage '" + stage + "'")
        print(traceback.format_exc())


def run_ga(json_dnn_path, json_ga_conf_path, parr_threads, output_file_path):
    """
    Run max memory save (mms) GA
    :param json_dnn_path: path to DNN model saved in .json format
    :param json_ga_conf_path: path to json GA config for MMS GA
    :param parr_threads: parallel CPU threads to run GA on
    :param output_file_path: path to output json file, where the
     pareto-front of MMS-GA chromosomes, delivered by MMS-GA will be saved
    """
    from converters.json_converters.json_to_dnn import parse_json_dnn
    from converters.json_converters.json_mms_ga_conf_parser import parse_mms_ga_conf
    from converters.json_converters.mms_chromosomes_to_json import mms_chromosomes_to_json
    import traceback
    from util import print_to_stderr
    from DSE.low_memory.mms.ga_based.multi_thread.MMSgaParallel import MMSgaParallel

    stage = "DNN parsing"
    try:
        dnn = parse_json_dnn(json_dnn_path)
        # dnn.print_details()

        stage = "GA config parsing"
        conf = parse_mms_ga_conf(json_ga_conf_path)
        # print(conf)

        stage = "GA setup"
        ga = MMSgaParallel(dnn,
                           conf["epochs"],
                           conf["population_start_size"],
                           conf["selection_percent"],
                           conf["mutation_probability"],
                           conf["mutation_percent"],
                           conf["max_no_improvement_epochs"],
                           conf["dp_by_parts_init_probability"],
                           conf["data_token_size"],
                           parr_threads,
                           conf["verbose"])

        stage = "GA initialization with first population"
        ga.init_with_random_population()

        # for chromosome in ga.population:
        #    chromosome.print_long()

        stage = "GA execution"
        pareto_front = ga.run()

        # print("GA returned pareto front of", len(pareto_front), "elements, with min-buffers chromosome:")
        # best (in terms of buffers sizes) chromosome
        # best_chromosome = pareto_front[0]
        # print(best_chromosome.dp_by_parts)

        stage = "Output JSON file generation"
        mms_chromosomes_to_json(pareto_front, output_file_path)

    except Exception:
        print_to_stderr("Exception at stage '" + stage + "'")
        print(traceback.format_exc())

