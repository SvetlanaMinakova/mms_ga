import json
from converters.json_converters.json_util import extract_or_default


def parse_mms_ga_conf(path):
    """ Parse max-memory-save (MMS) GA config """
    with open(path, 'r') as file:
        if file is None:
            raise FileNotFoundError
        else:
            conf_as_dict = {}
            conf = json.load(file)
            conf_as_dict["epochs"] = extract_or_default(conf, "epochs", 10)
            conf_as_dict["population_start_size"] = extract_or_default(conf, "population_start_size", 100)
            conf_as_dict["selection_percent"] = extract_or_default(conf, "selection_percent", 50)
            conf_as_dict["mutation_probability"] = extract_or_default(conf, "mutation_probability", 0.0)
            conf_as_dict["mutation_percent"] = extract_or_default(conf, "mutation_percent", 10)
            conf_as_dict["max_no_improvement_epochs"] = extract_or_default(conf, "max_no_improvement_epochs", 10)
            conf_as_dict["dp_by_parts_init_probability"] = extract_or_default(conf, "dp_by_parts_init_probability", 0.5)
            conf_as_dict["data_token_size"] = extract_or_default(conf, "data_token_size", 4)
            conf_as_dict["verbose"] = extract_or_default(conf, "verbose", True)
            return conf_as_dict

