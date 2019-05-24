import gzip
import tempfile
import unittest

from summarize.data.io.util import is_gz_file


class TestUtil(unittest.TestCase):
    def test_is_gz_file(self):
        with tempfile.NamedTemporaryFile() as temp:
            # Write a plain text file
            with open(temp.name, 'w') as out:
                out.write('plain text')
            assert is_gz_file(temp.name) is False

            # Write a gzipped file
            with gzip.open(temp.name, 'wb') as out:
                out.write(b'gzipped')
            assert is_gz_file(temp.name) is True
