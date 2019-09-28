python -m summarize.data.dataset_setup.cnn_dailymail data/cnn-dailymail
for split in train valid test; do
  for dataset in cnn dailymail cnn-dailymail; do
    python -m summarize.data.dataset_setup.tokenize \
      data/cnn-dailymail/${dataset}/${split}.jsonl.gz \
      data/cnn-dailymail/${dataset}/${split}.tokenized.jsonl.gz \
      document summary \
      --backend nltk
  done
done
