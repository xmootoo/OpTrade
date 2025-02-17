import os

def clean_up_file(
    file_path: str
) -> None:
    try:
        os.remove(file_path)
        print(f"Deleted file {file_path}")
    except OSError as e:
        print(f"Warning: Could not delete file {file_path}: {e}")
