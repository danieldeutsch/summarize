for split in train valid test; do
  python -m summarize.data.dataset_setup.deutsch2019 \
    data/wikicite/${split}.tokenized.v1.1.jsonl.gz \
    data/deutsch2019/${split}.v1.1.jsonl.gz \
    --num-cores 8
done
