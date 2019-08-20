expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir="${expt_dir}/output"
results_dir="${expt_dir}/results"
mkdir -p ${output_dir} ${results_dir}

date="2019-04-20"
max_words=200

for split in valid test; do
  gold_file="data/wikicite/${date}/preprocessed/${split}.jsonl.gz"
  model_file="${output_dir}/${split}.jsonl"
  metrics_file="${output_dir}/${split}.metrics.json"

  python -m summarize.utils.extract_cloze_from_labels \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.0.jsonl.gz \
    ${output_dir}/${split}.jsonl

  python -m summarize.metrics.rouge \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.0.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    --gold-summary-field-name cloze \
    --model-summary-field-name cloze \
    --add-wrapping-list \
    --compute-rouge-l \
    --silent \
    --output-file ${results_dir}/${split}.metrics.json
done
