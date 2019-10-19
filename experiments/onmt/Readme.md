# OpenNMT Parity Experiment
This experiment aims to compare the performance of the Summarize and OpenNMT models with as close to identical setups as possible to ensure parity between libraries.
The tests train and evaluate the sequence-to-sequence and pointer-generator models which are based on RNNs.
There is a directory for each model that includes more details and the specific commands to reproduce the results.
The OpenNMT commands come from the [summarization example](http://opennmt.net/OpenNMT-py/Summarization.html).

## OpenNMT Data Setup
The preprocessing of the CNN/DailyMail dataset is common between both OpenNMT models.
```
git clone https://github.com/OpenNMT/OpenNMT-py
cd OpenNMT-py
wget https://s3.amazonaws.com/opennmt-models/Summary/cnndm.tar.gz
mkdir data/cnndm
tar -xzvf cnndm.tar.gz -C data/cnndm

python preprocess.py \
  -train_src data/cnndm/train.txt.src \
  -train_tgt data/cnndm/train.txt.tgt.tagged \
  -valid_src data/cnndm/val.txt.src \
  -valid_tgt data/cnndm/val.txt.tgt.tagged \
  -save_data data/cnndm/CNNDM \
  -src_seq_length 10000 \
  -tgt_seq_length 10000 \
  -src_seq_length_trunc 400 \
  -tgt_seq_length_trunc 100 \
  -dynamic_dict \
  -share_vocab \
  -shard_size 100000
```
