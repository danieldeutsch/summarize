for split in train valid test; do
  python -m summarize.data.dataset_setup.deutsch2019 \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/${split}.tokenized.v1.0.jsonl.gz \
    data/deutsch2019/${split}.jsonl.gz
done
