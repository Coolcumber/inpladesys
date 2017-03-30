from typing import List
import os.path


def get_files(directory: str, filter) -> List[str]:
    """ Returns a sorted list of full paths of files in the directory. """
    files = [f for f in (os.path.join(directory, e) for e in os.listdir(directory)) if os.path.isfile(f) & filter(f)]
    files.sort()
    return files

