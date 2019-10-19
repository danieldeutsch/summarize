expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
model_file=${expt_dir}/model/model.tar.gz
output_dir=${expt_dir}/output

mkdir -p ${output_dir}

for split in valid test; do
  # add minimum length
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.min-length.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --use-dataset-reader \
    --overrides '{"model.beam_search.disallow_repeated_ngrams": null, "model.beam_search.repeated_ngrams_exceptions": null, "model.beam_search.length_penalizer": null, "model.beam_search.coverage_penalizer": null}' \
    ${model_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/onmt/${split}.v1.0.jsonl.gz

  # add disallow repeated trigrams
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.repeated-trigrams.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --use-dataset-reader \
    --overrides '{"model.beam_search.length_penalizer": null, "model.beam_search.coverage_penalizer": null}' \
    ${model_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/onmt/${split}.v1.0.jsonl.gz

  # add length penalizer
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.length.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --use-dataset-reader \
    --overrides '{"model.beam_search.coverage_penalizer": null}' \
    ${model_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/onmt/${split}.v1.0.jsonl.gz

  # add coverage penalizer
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.coverage.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --use-dataset-reader \
    ${model_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/onmt/${split}.v1.0.jsonl.gz
done
