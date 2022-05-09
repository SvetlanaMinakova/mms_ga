def select_best_chromosome(json_chromosomes: [], max_buf_size_mb=-1, max_latency_loss_ms=-1):
    """
    Select best chromosome from a pareto front
    :param json_chromosomes: pareto-front, i.e., list of chromosomes,
        where every chromosome is represented as json-dictionary
    :param max_buf_size_mb: memory constraint, i.e., maximum amount of memory (in megaBytes)
        occupied by the application buffers. If = -1, an application does not have a memory constraint
    :param max_latency_loss_ms: latency loss constraint, i.e., maximum amount of execution time
        delay (latency) introduced into application by the DNN-based application memory reduction.
        If -1, the application does not have the latency loss constraint
    :return: list of chromosomes, represented as json-dictionary, where every chromosome meets
        memory cost and latency loss constraints
    """
    pass


def filter_chromosomes(json_chromosomes: [], max_buf_size_mb=-1, max_latency_loss_ms=-1):
    """
    Filter json chromosomes and return only those chromosomes that meet memory cost and latency loss constraints
    :param json_chromosomes: list of chromosomes, where every chromosome is represented as json-dictionary
    :param max_buf_size_mb: memory constraint, i.e., maximum amount of memory (in megaBytes)
        occupied by the application buffers. If = -1, an application does not have a memory constraint
    :param max_latency_loss_ms: latency loss constraint, i.e., maximum amount of execution time
        delay (latency) introduced into application by the DNN-based application memory reduction.
        If -1, the application does not have the latency loss constraint
    :return: list of chromosomes, represented as json-dictionary, where every chromosome meets
        memory cost and latency loss constraints
    """
    json_chromosomes_filtered = []
    for json_chromosome in json_chromosomes:
        if max_buf_size_mb == -1 or json_chromosome["buf_size"] <= max_buf_size_mb:
            if max_latency_loss_ms == -1 or json_chromosome["time_loss"] <= max_latency_loss_ms:
                json_chromosomes_filtered.append(json_chromosome)
    return json_chromosomes_filtered


def select_chromosome_with_min_time_loss(json_chromosomes: []):
    """
    From list of chromosomes choose chromosome with minimum time loss
    :param json_chromosomes: list of chromosomes, where every chromosome is represented as json-dictionary
    :return: chromosome with minimum time loss
    """
    if len(json_chromosomes) < 1:
        raise Exception("Chromosomes selection error: chromosomes list is empty")
    sorted_chromosomes = sorted(json_chromosomes, key=lambda elem: elem["time_loss"])
    return sorted_chromosomes[0]


def select_chromosome_with_min_buf_size(json_chromosomes: []):
    """
    From list of chromosomes choose chromosome with minimum buffers size
    :param json_chromosomes: list of chromosomes, where every chromosome is represented as json-dictionary
    :return: chromosome with minimum time loss
    """
    if len(json_chromosomes) < 1:
        raise Exception("Chromosomes selection error: chromosomes list is empty")
    sorted_chromosomes = sorted(json_chromosomes, key=lambda elem: elem["buf_size"])
    return sorted_chromosomes[0]


def print_pareto(ga_json_output_path):
    """
    Print pareto front
    :param ga_json_output_path: path to the pareto front
    saved in the .json file
    :return: chromosome
    """
    from converters.json_converters.json_to_ga_result import parse_json_ga_result
    chromosomes = parse_json_ga_result(ga_json_output_path)
    delay_per_phase_ms = 0.0005
    buf_size_round = 4
    # sort chromosomes by buffer size (in MB)
    chromosomes = sorted(chromosomes, key=lambda x: x.buf_size, reverse=False)
    print("BUF SIZE (MB); PHASES")
    for chromosome in chromosomes:
        buf_size = chromosome.buf_size
        phases = int(chromosome.time_loss/delay_per_phase_ms)
        print(round(buf_size, buf_size_round), ";", phases)