import bz2
import gzip
import json
import tempfile
import unittest

from summarize.data.io import JsonlReader


class TestJsonlReader(unittest.TestCase):
    def setUp(self):
        self.data = [
            {'a': 4, 'b': 'testing'},
            {'c': [1, 2, 3]}
        ]

    def test_plain_file(self):
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl')
        with open(temp_file.name, 'w') as out:
            for item in self.data:
                serialzed = json.dumps(item)
                out.write(serialzed + '\n')

        # Load from file, ensure it is correct
        actual_data = []
        with JsonlReader(temp_file.name) as f:
            for item in f:
                actual_data.append(item)
        self.assertEqual(self.data, actual_data)

    def test_gzip_file(self):
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl.gz')
        with gzip.open(temp_file.name, 'wb') as out:
            for item in self.data:
                serialzed = json.dumps(item).encode()
                out.write(serialzed + b'\n')

        # Load from file, ensure it is correct
        actual_data = []
        with JsonlReader(temp_file.name) as f:
            for item in f:
                actual_data.append(item)
        self.assertEqual(self.data, actual_data)

    def test_gzip_file_no_extension(self):
        """Tests a gzip file that does not have a ".gz" extension."""
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile()
        with gzip.open(temp_file.name, 'wb') as out:
            for item in self.data:
                serialzed = json.dumps(item).encode()
                out.write(serialzed + b'\n')

        # Load from file, ensure it is correct
        actual_data = []
        with JsonlReader(temp_file.name) as f:
            for item in f:
                actual_data.append(item)
        self.assertEqual(self.data, actual_data)

    def test_bz2_file(self):
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl.bz2')
        with bz2.open(temp_file.name, 'wb') as out:
            for item in self.data:
                serialzed = json.dumps(item).encode()
                out.write(serialzed + b'\n')

        # Load from file, ensure it is correct
        actual_data = []
        with JsonlReader(temp_file.name) as f:
            for item in f:
                actual_data.append(item)
        self.assertEqual(self.data, actual_data)

    def test_read(self):
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl')
        with open(temp_file.name, 'w') as out:
            for item in self.data:
                serialzed = json.dumps(item)
                out.write(serialzed + '\n')

        # Load from file, ensure it is correct
        actual_data = JsonlReader(temp_file.name).read()
        self.assertEqual(self.data, actual_data)
