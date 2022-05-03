from DSE.low_memory.mms.ga_based.MMSChromosome import MMSChromosome
from converters.json_converters.JSONNestedClassVisitor import JSONNestedClassVisitor
from models.dnn_model.dnn import DNN


def mms_chromosomes_to_json(chromosomes: [MMSChromosome], filepath: str):
    """
    Convert MMS GA chromosome into a JSON string and saves in output file
    :param chromosomes: MMS
    :param filepath: path to target .json file
    :return: JSON string, encoding the analytical DNN model
    """
    visitor = MMSChromosomeJSONNestedClassVisitor(chromosomes, filepath)
    visitor.run()


class MMSChromosomeJSONNestedClassVisitor (JSONNestedClassVisitor):
    """
    JSON visitor. Used to save custom classes as a .json file.
    Attributes:
        chromosomes: list of MMS GA chromosomes (each is an object of MMSChromosome class)
        filepath: path to target .json file
    """
    def __init__(self, chromosomes: [MMSChromosome], filepath):
        super(MMSChromosomeJSONNestedClassVisitor, self).__init__(chromosomes, filepath)

    def visit_object(self, obj):
        """
        Recursive object visitor
        :param obj: object to visit
        """
        # do not visit dnn
        if isinstance(obj, DNN):
            return

        super().visit_object(obj)

