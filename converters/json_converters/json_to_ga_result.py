from low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from converters.json_converters.json_util import extract_or_default, parse_list


def parse_json_ga_result(path):
    """
    Converts a JSON File into an (analytical) DNN model
    :param path: path to .json file
    """
    chromosomes_desc_list = parse_list(path)
    chromosomes = []
    for chromosome_desc in chromosomes_desc_list:
        # read fields
        layers_num = chromosome_desc["layers_num"]
        dp_by_parts = chromosome_desc["dp_by_parts"]
        time_loss = chromosome_desc["time_loss"]
        buf_size = chromosome_desc["buf_size"]

        # init chromosome
        chromosome = MMSChromosome(layers_num)
        chromosome.dp_by_parts = dp_by_parts
        chromosome.time_loss = time_loss
        chromosome.buf_size = buf_size
        # add chromosome to the list
        chromosomes.append(chromosome)
    return chromosomes


