import os
import shutil

def clean_up_file(
    file_path: str
) -> None:
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"Warning: Could not delete file {file_path}: {e}")


def clean_up_dir(
    dir_path: str
) -> None:
    """
    Remove a directory and all its contents (files and subdirectories).

    Args:
        dir_path (str): Path to the directory to be removed
    """
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print(f"Warning: Could not delete directory {dir_path}: {e}")
