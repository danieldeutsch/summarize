expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir="${expt_dir}/output"
results_dir="${expt_dir}/results"
mkdir -p ${output_dir} ${results_dir}

model="345M"

for split in valid test; do
  python -m summarize.models.cloze.open_ai_language_model \
    ${expt_dir}/models/${model} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    1 \
    40

  python -m summarize.metrics.rouge \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    --gold-summary-field-name cloze \
    --model-summary-field-name cloze \
    --add-gold-wrapping-list \
    --add-model-wrapping-list \
    --silent \
    --output-file ${results_dir}/${split}.metrics.json
done
