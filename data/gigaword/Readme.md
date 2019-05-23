# Gigaword
## Setup
To setup the Gigaword corpus, run the following command:
```
python -m summarize.data.dataset_setup.gigaword \
  data/gigaword
```
The script downloads the data from https://github.com/harvardnlp/sent-summary, replaces the `UNK` token with the AllenNLP special token for out-of-vocabulary words, and saves the data in the jsonl format.

There are 3,803,957 training, 189,651, and 1951 testing examples.

This is the dataset which is used to train the [OpenNMT-py Gigaword summarization models](http://opennmt.net/Models-py/#summarization).
I assume it is also the data used by [Rush et al. (2015)](https://www.aclweb.org/anthology/D15-1044), but the paper does not link to any dataset, code, or specify the size of the datasets splits.
The follow up work, [Ranzato et al. (2016)](https://arxiv.org/pdf/1511.06732.pdf), also uses Gigaword, but the dataset split sizes are very different (179,414 training, 22,568 validation, and 22,259 testing examples).
The [corresponding repository](https://github.com/facebookarchive/MIXER) only has instructions and code for the machine translation experiments.
