import os


def files_exist(files_abs_paths: []):
    """
    Check if every file in the list exists
    :param files_abs_paths list of abs paths to files
    :return: True if all files in the list exist, False otherwise
    """
    for file_path in files_abs_paths:
        if not file_exists(file_path):
            return False
    return True


def file_exists(file_abs_path):
    """
    Check if a file exists
    :param file_abs_path abs paths to the filef
    :return: True if file exists, False otherwise
    """
    f_exists = os.path.exists(file_abs_path)
    return f_exists


def create_dir_if_does_not_exist(dir_path: str):
    """
    if the directory is not present, create it.
    :param dir_path: path to directory
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def clear_folder(directory_to_clear):
    """
    Recursively delete directory with files
    :param directory_to_clear: path to directory with files
    """
    if os.path.exists(directory_to_clear):
        for the_file in os.listdir(directory_to_clear):
            file_path = os.path.join(directory_to_clear, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                else:
                    clear_folder(file_path)
                    os.rmdir(file_path)
            except Exception as e:
                print(e)


def create_or_overwrite_dir(dir_path):
    """
    Delete and re-create directory
    :param dir_path: directory path
    """
    if os.path.exists(dir_path):
        clear_folder(dir_path)
    else:
        os.makedirs(dir_path)


def file_paths_in_directory(directory, extensions=None):
    """
    Find all file paths in given directory
    :param directory: directory
    :param extensions: expected file extensions (filter).
        If extensions are specified as None, files of any extension are considered.
        Otherwise, only files that have specified extensions are considered.
    :return: list of  file paths
    """
    file_paths = []
    if os.path.exists(directory):
        for the_file in os.listdir(directory):
            file_path = os.path.join(directory, the_file)
            if os.path.isfile(file_path):
                if file_has_extension(file_path, extensions):
                    file_paths.append(str(file_path))
        return file_paths
    else:
        raise Exception("Directory " + directory + "does not exist")


def file_has_extension(file_path, extensions=None):
    """
    Check if file has one of expected extensions
    :param file_path: path to the file
    :param extensions: expected file extensions (filter).
        If extensions are specified as None, any file passes the filter
    :return: True, if file has one of the expected extensions or
        if extensions are unspecified and False otherwise
    """
    if extensions is None:
        return True
    for extension in extensions:
        file_postfix = "." + extension
        if str(file_path).endswith(file_postfix):
            return True
    return False
