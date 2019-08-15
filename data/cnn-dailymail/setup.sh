python -m summarize.data.dataset_setup.cnn_dailymail data/cnn-dailymail
for split in train valid test; do
  python -m summarize.data.dataset_setup.tokenize \
    data/cnn-dailymail/cnn-dailymail/${split}.jsonl.gz \
    data/cnn-dailymail/cnn-dailymail/${split}.tokenized.jsonl.gz \
    document summary \
    --backend nltk
done
