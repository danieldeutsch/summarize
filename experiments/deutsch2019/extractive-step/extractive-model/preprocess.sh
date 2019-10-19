if [ "$#" -ne 2 ]; then
    echo "Usage: sh preprocess.sh <use-topics> <use-context>"
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
model_file=${expt_dir}/model/${topics_dir}/${context_dir}/model.tar.gz
preprocess_dir=${expt_dir}/preprocessed/${topics_dir}/${context_dir}
mkdir -p ${preprocess_dir}

for split in train valid test; do
  temp_file=$(mktemp)
  allennlp predict \
    --include-package summarize \
    --predictor cloze-extractive-predictor \
    --output-file ${temp_file} \
    --cuda-device 0 \
    --batch-size 1 \
    --silent \
    --use-dataset-reader \
    --overrides '{"dataset_reader.max_num_sentences": null}' \
    ${model_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz

  python -m summarize.utils.copy_jsonl_fields \
    ${temp_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${preprocess_dir}/${split}.jsonl.gz \
    --field-names cloze document

  rm ${temp_file}
done
