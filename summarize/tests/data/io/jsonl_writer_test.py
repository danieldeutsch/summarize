import bz2
import gzip
import json
import tempfile
import unittest

from summarize.data.io import JsonlWriter


class TestJsonlWriter(unittest.TestCase):
    def setUp(self):
        self.data = [
            {'a': 4, 'b': 'testing'},
            {'c': [1, 2, 3]}
        ]

    def test_plain_file(self):
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl')
        with JsonlWriter(temp_file.name) as out:
            for item in self.data:
                out.write(item)

        # Load from file, ensure it is correct
        actual_data = []
        with open(temp_file.name, 'r') as f:
            for line in f:
                actual_data.append(json.loads(line))
        self.assertEqual(self.data, actual_data)

    def test_gzip_file(self):
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl.gz')
        with JsonlWriter(temp_file.name) as out:
            for item in self.data:
                out.write(item)

        # Load from file, ensure it is correct
        actual_data = []
        with gzip.open(temp_file.name, 'rb') as f:
            for line in f:
                actual_data.append(json.loads(line.decode()))
        self.assertEqual(self.data, actual_data)

    def test_bz2_file(self):
        # Write the data to a file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jsonl.bz2')
        with JsonlWriter(temp_file.name) as out:
            for item in self.data:
                out.write(item)

        # Load from file, ensure it is correct
        actual_data = []
        with bz2.open(temp_file.name, 'rb') as f:
            for line in f:
                actual_data.append(json.loads(line.decode()))
        self.assertEqual(self.data, actual_data)
