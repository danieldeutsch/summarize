This directory contains the script to preprocess the WikiCite dataset for "Summary Cloze: A New Task for Content Selection in Topic-Focused Summarization."

First, run the setup script under the `data/wikicite` directory.
Then, run the `setup.sh` script to compute the ROUGE-based heuristic extractive labels for the dataset.
The processing speed is somewhat slow, so it may take several hours to process the data.
Alternatively, the preprocessed data can be downloaded here:
<a href="https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/train.v1.0.jsonl.gz">train</a>,
<a href="https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/valid.v1.0.jsonl.gz">valid</a>,
<a href="https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/test.v1.0.jsonl.gz">test</a>.
