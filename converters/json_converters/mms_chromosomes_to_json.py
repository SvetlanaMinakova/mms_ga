from DSE.low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from converters.json_converters.JSONNestedClassVisitor import JSONNestedClassVisitor
from models.dnn_model.dnn import DNN
from fileworkers.json_fw import save_as_json


def mms_chromosomes_to_json(chromosomes: [MMSChromosome], filepath: str):
    """
    Convert MMS GA chromosome into a JSON string and saves in output file
    :param chromosomes: MMS
    :param filepath: path to target .json file
    :return: JSON string, encoding the analytical DNN model
    """
    # visitor = MMSChromosomeJSONNestedClassVisitor(chromosomes, filepath)
    # visitor.run()
    json_chromosomes = []

    for chromosome in chromosomes:
        json_chromosome = {
            "layers_num": chromosome.layers_num,
            "dp_by_parts": chromosome.dp_by_parts,
            "time_loss": chromosome.time_loss,
            "buf_size": chromosome.buf_size
        }
        json_chromosomes.append(json_chromosome)

    chromosomes_as_dict = {"chromosomes": json_chromosomes}
    save_as_json(filepath, chromosomes_as_dict)

