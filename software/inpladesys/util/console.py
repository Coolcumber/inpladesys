import select
import sys


def read_line(wait: bool = True, skip_to_last_line: bool = False):
    """
    Returns a line from standard input.
    :param wait: If set to False and no input is available, returns None.
    :param skip_to_last_line: If set to True and more lines are available, consumes all lines and returns the last.
    """
    def input_available():
        return select.select([sys.stdin], [], [], 0)[0]

    cmd = None
    if wait or input_available():
        cmd = input()
        if skip_to_last_line:
            while input_available():
                cmd = input()
    return cmd
