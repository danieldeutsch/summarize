"""
Replaces the model configuration in a model.tar.gz file with a new one. The
new configuration can be a jsonnet file that will be evaluated into a json.
"""
import argparse
import tarfile
import json
from io import BytesIO

from allennlp.common.params import Params


def main(args):
    tar_bytes = open(args.model_file_path, 'rb').read()
    with tarfile.open(fileobj=BytesIO(tar_bytes), mode='r:gz') as tar:
        with tarfile.open(args.output_file_path, 'w:gz') as out:
            for member in tar.getmembers():
                if member.name != 'config.json':
                    out.addfile(member, tar.extractfile(member.name))
                else:
                    new_params = Params.from_file(args.config_file_path)
                    serialized_params = json.dumps(new_params.as_ordered_dict(), indent=4).encode()
                    bytes_io = BytesIO(serialized_params)
                    member.size = len(serialized_params)
                    out.addfile(tarinfo=member, fileobj=bytes_io)


if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('model_file_path', help='The path to the model.tar.gz with the config to replace')
    argp.add_argument('output_file_path', help='The path to the new model.tar.gz')
    argp.add_argument('config_file_path', help='The path to the new config file')
    args = argp.parse_args()
    main(args)
