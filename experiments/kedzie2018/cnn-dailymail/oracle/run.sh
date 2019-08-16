expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir="${expt_dir}/output"
results_dir="${expt_dir}/results"
mkdir -p ${output_dir}

for split in valid test; do
  python -m summarize.utils.extract_summary_from_labels \
    data/kedzie2018/cnn-dailymail/${split}.jsonl.gz \
    output/${split}.jsonl

  python -m summarize.metrics.rouge \
    data/kedzie2018/cnn-dailymail/${split}.jsonl.gz \
    output/${split}.jsonl \
    --silent \
    --max-ngram 2 \
    --remove-stopwords \
    --max-words 100 \
    --output-file ${results_dir}/${split}.metrics.json
done
