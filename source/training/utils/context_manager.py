"""
This module provides custom context managers for common tasks
such as output redirection and temporary directory management.
These utilities help in creating cleaner and more robust code by encapsulating setup and teardown logic.

- `TeeStdout`: Redirects `sys.stdout` to both the console and a file, useful for logging.
- `OptionalTempDir`: Manages a directory that can either be a user-specified path or
  a temporary directory that is automatically cleaned up on exit.
"""

import os
import sys
import shutil
import tempfile
import contextlib

from datetime import datetime
from typing_extensions import override, Optional



class TeeStdout(contextlib.AbstractContextManager):
    """
    A context manager to redirect `sys.stdout` to both the console and a file.

    This is useful for capturing the full output of a script to a log file while
    still seeing the output in real-time on the terminal.
    A timestamp is added to the file upon entry to delineate different runs.

    Usage:
        with TeeStdout('logfile.txt'):
            print("This will be shown on screen and saved to logfile.txt")
    """

    def __init__(self, filename, mode="a"):
        """
        Initializes the TeeStdout context manager.

        :param filename: The path to the file where stdout will be saved.
                         The directory will be created if it does not exist.
        :param mode: The file opening mode (e.g., 'a' for append, 'w' for write).
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file = open(filename, mode)
        self.stdout = sys.stdout

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]  # trims microseconds to milliseconds
        self.file.write(f"\n\n---------------- {now} ----------------\n\n")
        self.file.flush()


    def write(self, data):
        """
        Writes data to both the original stdout (console) and the file.

        :param data: The string data to write.
        """
        self.stdout.write(data)   # Print to terminal
        self.file.write(data)     # Save to file


    def flush(self):
        """Flushes both the stdout and file streams."""
        self.stdout.flush()
        self.file.flush()


    @override
    def __enter__(self):
        """Redirects `sys.stdout` to this instance."""
        sys.stdout = self
        return self


    @override
    def __exit__(self, exc_type, exc_value, traceback):
        """Restores the original `sys.stdout` and closes the file."""
        sys.stdout = self.stdout
        self.file.close()



class OptionalTempDir(contextlib.AbstractContextManager):
    """
    A context manager that provides a directory path.

    If a directory path is supplied during initialization,
    that path is used and is not cleaned up on exit.
    If no path is supplied, a temporary directory is created
    and will be automatically removed when the context is exited.

    This is useful for functions that need a place to write files,
    where the user may want to either keep the files in a specific location
    or treat them as temporary.

    Usage:
        # With a user-supplied directory (not deleted)
        with OptionalTempDir(supplied_dir='/path/to/my/dir') as d:
            # ... write files to d ...

        # With a temporary directory (deleted on exit)
        with OptionalTempDir() as d:
            # ... write files to d ...
    """

    def __init__(self, supplied_dir: Optional[str] = None, prefix: Optional[str] = None):
        """
        Initializes the OptionalTempDir context manager.

        :param supplied_dir: An optional path to a directory.
                             If provided, this directory will be used and will not be deleted on exit.
                             The directory will be created if it doesn't exist.
        :param prefix: An optional prefix for the temporary directory name,
                       used only if `supplied_dir` is None.
        """
        if supplied_dir is not None: os.makedirs(supplied_dir, exist_ok=True)
        self.supplied_dir = supplied_dir
        self.prefix = prefix
        self.temp_dir = None


    @override
    def __enter__(self):
        """
        Enters the context, returning the path to the directory.

        :return: The path to the user-supplied directory or the new temporary directory.
        """
        if self.supplied_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix=self.prefix)
            return self.temp_dir
        else:
            return self.supplied_dir


    @override
    def __exit__(self, exc_type, exc_value, traceback):
        """Exits the context, removing the temporary directory if one was created."""
        if self.temp_dir is not None:
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Temporary model directory {self.temp_dir} has been deleted.")
            except Exception as e:
                print(f"Failed to remove temporary model directory {self.temp_dir}: {e}")
