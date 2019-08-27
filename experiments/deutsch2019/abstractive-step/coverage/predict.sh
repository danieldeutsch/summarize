expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [ "$#" -ne 2 ]; then
    echo "Usage: sh predict.sh <preprocessing-dataset> <use-context>"
    exit
fi

preprocessing_dataset=$1
use_context=$2
if [ "${preprocessing_dataset}" == "lead" ]; then
  preprocess_dir="${expt_dir}/../../extractive-step/lead/preprocessed"
elif [ "${preprocessing_dataset}" == "oracle" ]; then
  preprocess_dir="${expt_dir}/../../extractive-step/oracle/preprocessed"
elif [ "${preprocessing_dataset}" == "extractive-model" ]; then
  preprocess_dir="${expt_dir}/../../extractive-step/extractive-model/preprocessed/topics/context"
else
  echo "Invalid preprocessing dataset: ${preprocessing_dataset}"
  exit
fi

if [ "${use_context}" == "true" ]; then
  context_dir="context"
else
  context_dir="no-context"
fi

model_dir=${expt_dir}/model/${preprocessing_dataset}/${context_dir}
model_file=${model_dir}/model.tar.gz
output_dir=${expt_dir}/output/${preprocessing_dataset}/${context_dir}
mkdir -p ${output_dir}

for split in valid test; do
  allennlp predict \
    --include-package summarize \
    --output-file ${output_dir}/${split}.jsonl \
    --predictor cloze-abstractive-predictor \
    --silent \
    --use-dataset-reader \
    --cuda-device 0 \
    --batch-size 16 \
    ${model_file} \
    ${preprocess_dir}/${split}.jsonl.gz
done
