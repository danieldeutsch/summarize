if [ "$#" -ne 2 ]; then
    echo "Usage: sh evaluate.sh <use-topics> <use-context>"
    exit
fi

use_topics=$1
use_context=$2
if [ "${use_topics}" == "true" ]; then
  topics_dir="topics"
else
  topics_dir="no-topics"
fi
if [ "${use_context}" == "true" ]; then
  context_dir="context"
else
  context_dir="no-context"
fi

expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir=${expt_dir}/output/${topics_dir}/${context_dir}
results_dir=${expt_dir}/results/${topics_dir}/${context_dir}
mkdir -p ${results_dir}

for split in valid test; do
  python -m summarize.metrics.rouge \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${output_dir}/${split}.max-tokens.jsonl \
    --gold-summary-field-name cloze \
    --model-summary-field-name cloze \
    --add-gold-wrapping-list \
    --compute-rouge-l \
    --silent \
    --max-words 200 \
    --output-file ${results_dir}/${split}.max-tokens.metrics.json

  python -m summarize.metrics.rouge \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${output_dir}/${split}.max-sents.jsonl \
    --gold-summary-field-name cloze \
    --model-summary-field-name cloze \
    --add-gold-wrapping-list \
    --compute-rouge-l \
    --silent \
    --output-file ${results_dir}/${split}.max-sents.metrics.json
done
