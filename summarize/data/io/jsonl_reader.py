import bz2
import gzip
import json
from typing import Any

from summarize.data.io.util import is_gz_file


class JsonlReader(object):
    """
    The `JsonlReader` is a layer of abstraction around reading serialized
    objects from a jsonl file. The reader will automatically deserialize and return
    one object from each line in the file. The data in the file will be decoded
    from a binary file depending on the extension of the file name. Current
    supported binary formats are gzip (``.gz``) and bz2 (``.bz2``). For gzip only,
    this will also inspect the file to see if it's gzipped in addition to checking
    the extension.

    The class should be used the same way that a built-in file handler works::

        with JsonlReader('/path/to/file.jsonl.gz') as f:
            for data in f:
                ...

    Parameters
    ----------
    file_path: ``str``
        The path to the file where the data should be read.
    """
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path

    def __enter__(self):
        self.binary = False
        if self.file_path.endswith('.gz') or is_gz_file(self.file_path):
            self.file_handler = gzip.open(self.file_path, 'rb')
            self.binary = True
        elif self.file_path.endswith('.bz2'):
            self.file_handler = bz2.open(self.file_path, 'rb')
            self.binary = True
        else:
            self.file_handler = open(self.file_path, 'r')
            self.binary = False
        return self

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        for line in self.file_handler:
            if self.binary:
                line = line.decode()
            return json.loads(line)
        raise StopIteration

    def __exit__(self, *args):
        self.file_handler.close()
