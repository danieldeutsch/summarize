expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir="${expt_dir}/output"
results_dir="${expt_dir}/results"
mkdir -p ${output_dir} ${results_dir}

for split in valid test; do
  python -m summarize.models.cloze.lead \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.0.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    --max-tokens 200

  python -m summarize.metrics.rouge \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.0.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    --gold-summary-field-name cloze \
    --model-summary-field-name cloze \
    --add-wrapping-list \
    --compute-rouge-l \
    --silent \
    --max-words 200 \
    --output-file ${results_dir}/${split}.metrics.json
done
