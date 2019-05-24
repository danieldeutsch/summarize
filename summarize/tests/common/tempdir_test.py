import os
import shutil
import unittest

from summarize.common import TemporaryDirectory


class TestTemporaryDirectory(unittest.TestCase):
    def test_temporary_directory(self):
        with TemporaryDirectory() as temp_dir:
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)
        assert not os.path.exists(temp_dir)

    def test_temporary_directory_root(self):
        # Create two temporary directories with one inside the other
        # to make sure it was created in the correct location
        with TemporaryDirectory() as root_temp_dir:
            with TemporaryDirectory(root=root_temp_dir) as temp_dir:
                assert os.path.exists(temp_dir)
                assert os.path.isdir(temp_dir)
                assert temp_dir.startswith(root_temp_dir)

    def test_temporary_directory_persist(self):
        with TemporaryDirectory(persist=True) as temp_dir:
            assert os.path.exists(temp_dir)
            assert os.path.isdir(temp_dir)
        assert os.path.exists(temp_dir)
        shutil.rmtree(temp_dir)
        assert not os.path.exists(temp_dir)
