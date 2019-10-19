expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir="${expt_dir}/output"
results_dir="${expt_dir}/results"
mkdir -p ${output_dir} ${results_dir}

for split in valid test; do
  for metric in "R1-F1" "R2-F1" "RL-F1"; do
    python -m summarize.models.cloze.oracle \
      https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
      ${output_dir}/${split}.${metric}.jsonl \
      ${metric} \
      --max-sentences 1 \
      --cloze-only

    python -m summarize.metrics.rouge \
      https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
      ${output_dir}/${split}.${metric}.jsonl \
      --gold-summary-field-name cloze \
      --model-summary-field-name cloze \
      --add-gold-wrapping-list \
      --add-model-wrapping-list \
      --compute-rouge-l \
      --silent \
      --output-file ${results_dir}/${split}.${metric}.metrics.json
  done
done
