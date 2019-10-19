if [ "$#" -ne 2 ]; then
    echo "Usage: sh predict.sh <use-topics> <use-context>"
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
output_dir=${expt_dir}/output/${topics_dir}/${context_dir}
mkdir -p ${output_dir}

for split in valid test; do
  allennlp predict \
    --include-package summarize \
    --predictor cloze-extractive-predictor \
    --output-file ${output_dir}/${split}.max-tokens.jsonl \
    --cuda-device 0 \
    --batch-size 1 \
    --silent \
    --use-dataset-reader \
    --overrides '{"dataset_reader.max_num_sentences": null}' \
    ${model_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz

  allennlp predict \
    --include-package summarize \
    --predictor cloze-extractive-predictor \
    --output-file ${output_dir}/${split}.max-sents.jsonl \
    --cuda-device 0 \
    --batch-size 1 \
    --silent \
    --use-dataset-reader \
    --overrides '{"dataset_reader.max_num_sentences": null, "model.max_words": null, "model.max_sents": 1}' \
    ${model_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz
done
