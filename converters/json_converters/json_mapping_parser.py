from converters.json_converters.json_util import parse_list


def parse_mapping(filepath):
    """
    Parse mapping of a DNN onto processors of target platform
    :param filepath: path to mapping, saved in .json file
    :return: mapping = [proc_tasks_1, proc_tasks_2, ..., proc_tasks_M]
    where M is number of processors in the target platform,
    proc_tasks_j = [task_j1, task_j2, ...], j in [1, M] is a set of tasks
    where task_ji is an integer number, representing layer (task) id in the DNN
    """
    mapping = parse_list(filepath)
    return mapping

