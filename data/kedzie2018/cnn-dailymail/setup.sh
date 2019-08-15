for split in train valid test; do
  python -m summarize.data.dataset_setup.kedzie2018 \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz \
    data/kedzie2018/cnn-dailymail/${split}.jsonl.gz \
    100 \
    --num-cores 16
done
