expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
model_file=${expt_dir}/model/model.tar.gz
output_dir=${expt_dir}/output

mkdir -p ${output_dir}

for split in valid test; do
  # No extra constraints
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.none.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --overrides '{"model.beam_search.min_steps": null, "model.beam_search.disallow_repeated_ngrams": null, "model.beam_search.length_penalizer": null, "model.beam_search.coverage_penalizer": null}' \
    ${model_file} \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz

  # add minimum length
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.min-length.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --overrides '{"model.beam_search.disallow_repeated_ngrams": null, "model.beam_search.length_penalizer": null, "model.beam_search.coverage_penalizer": null}' \
    ${model_file} \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz

  # add disallow repeated trigrams
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.repeated-trigrams.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --overrides '{"model.beam_search.length_penalizer": null, "model.beam_search.coverage_penalizer": null}' \
    ${model_file} \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz

  # add length penalizer
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.length.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --overrides '{model.beam_search.coverage_penalizer": null}' \
    ${model_file} \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz

  # add coverage penalizer
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.coverage.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    ${model_file} \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz
done
