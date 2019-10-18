for split in train valid test; do
  python -m summarize.data.dataset_setup.wikicite \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/${split}.v1.1.jsonl.gz \
    data/wikicite/${split}.tokenized.jsonl.gz
done
