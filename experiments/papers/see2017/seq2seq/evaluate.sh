expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir=${expt_dir}/output
results_dir=${expt_dir}/results

mkdir -p ${results_dir}

for split in valid test; do
  python -m summarize.metrics.rouge \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/${split}.tokenized.v1.0.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    --output-file ${results_dir}/${split}.metrics.json
done
