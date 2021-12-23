import json
"""
A bunch of common functions, useful for reading and parsing .json_converters files
"""


def extract_or_default(json_str, attr, default_value):
    """
    Extract value from json_converters string or return default value
    :param json_str: json_converters string
    :param attr: attribute name
    :param default_value: default value
    :return: extracted attribute value or default value
    """
    if attr in json_str:
        return json_str[attr]
    return default_value


def parse_list(filepath):
    """
    Parse list, specified in a .json_converters file
    :param filepath path to the .json_converters file
    :return list, specified in the .json_converters file
    """
    with open(filepath) as f:
        json_objs = json.load(f)
        return json_objs


