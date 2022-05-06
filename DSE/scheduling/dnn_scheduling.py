from enum import Enum


class DNNScheduling(Enum):
    """ Scheduling determines order in which CNN layers/partitions are executed"""
    # layers/partitions are executed one-by-one
    SEQUENTIAL = 1,
    # layers/partitions are executed as a pipeline (asap on respective processors)
    PIPELINE = 2,
    # custom, explicitly specified execution order
    CUSTOM = 3


def str_to_scheduling(str_scheduling: str):
    """ Convert string into DNNScheduling"""
    if str_scheduling.lower() == "sequential":
        return DNNScheduling.SEQUENTIAL
    if str_scheduling.lower() == "pipeline":
        return DNNScheduling.PIPELINE
    return DNNScheduling.CUSTOM

