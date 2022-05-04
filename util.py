import math
import sys
from pathlib import Path
import os

"""
Util - functions, shared among the project modules
"""


def get_project_root() -> Path:
    return Path(__file__).parent


def print_to_stderr(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def elements_prod(array):
    total_elements = 1
    for element in array:
        if element > 0:
            total_elements = total_elements * element
    return total_elements


def nth_root(x, n):
    root = x ** (1 / float(n))
    return root


def round_div(x, divider):
    div = math.ceil(float(x)/float(divider))
    return div


def giga():
    return 1e9


def mega():
    return 1e6


def milli():
    return 1e-3


def print_or_skip(txt, verbose):
    if verbose:
        print(txt)


def print_stage(stage: str, verbose):
    """ Helper-printout function for API scripts with per-stage verbose output"""
    if verbose:
        print("STAGE:", stage)

