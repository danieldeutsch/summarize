wget https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz -O data/onmt/cnndm.tar.gz
mkdir data/onmt/onmt
tar -xzvf data/onmt/cnndm.tar.gz -C data/onmt/onmt

python data/onmt/convert_to_jsonl.py \
  data/onmt/onmt/train.txt.src \
  data/onmt/onmt/train.txt.tgt.tagged \
  data/onmt/train.jsonl.gz

python data/onmt/convert_to_jsonl.py \
  data/onmt/onmt/val.txt.src \
  data/onmt/onmt/val.txt.tgt.tagged \
  data/onmt/valid.jsonl.gz

python data/onmt/convert_to_jsonl.py \
  data/onmt/onmt/test.txt.src \
  data/onmt/onmt/test.txt.tgt.tagged \
  data/onmt/test.jsonl.gz
