import json
import os


def read_json(abs_path):
    """
    Read json file
    :param abs_path: abs path to .json_converters file
    :return: json string, obtained from the file
    """
    with open(abs_path) as json_file:
        str_json = json.load(json_file)
    return str_json


def save_as_json(abs_path, data_json, pretty_printing=True):
    """
    Write json file
    :param abs_path: abs path to .json file
    :param pretty_printing: make .json printing pretty
    :param data_json json string to be written into the file
    """
    # create parent directory for file, if it doesn't exist
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    with open(abs_path, 'w') as f:
        if pretty_printing:
            json.dump(data_json, f, indent=4)
        else:
            json.dump(data_json, f)
