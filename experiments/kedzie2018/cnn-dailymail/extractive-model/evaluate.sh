if [ "$#" -ne 2 ]; then
    echo "Usage: sh evaluate.sh <encoder> <extractor>"
    exit
fi

encoder=$1
extractor=$2

expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir=${expt_dir}/output/${encoder}/${extractor}
results_dir=${expt_dir}/results/${encoder}/${extractor}

mkdir -p ${results_dir}

for split in valid test; do
  python -m summarize.metrics.rouge \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/kedzie2018/cnn-dailymail/${split}.v1.0.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    --silent \
    --max-ngram 2 \
    --max-words 100 \
    --remove-stopwords \
    --output-file ${results_dir}/${split}.metrics.json
done
