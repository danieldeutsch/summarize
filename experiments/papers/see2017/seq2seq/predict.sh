expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
model_file=${expt_dir}/model/model.tar.gz
output_dir=${expt_dir}/output

for split in valid test; do
  allennlp predict \
    --include-package summarize \
    --predictor sds-abstractive-predictor \
    --output-file ${output_dir}/${split}.jsonl \
    --silent \
    ${model_file} \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz
done
