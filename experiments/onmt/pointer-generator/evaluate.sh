expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir=${expt_dir}/output
results_dir=${expt_dir}/results

mkdir -p ${results_dir}

for split in valid test; do
  for constraints in none min-length repeated-trigrams length coverage; do
    python -m summarize.metrics.rouge \
      https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz \
      ${output_dir}/${split}.${constraints}.jsonl \
      --silent \
      --compute-rouge-l \
      --output-file ${results_dir}/${split}.${constraints}.metrics.json
  done
done