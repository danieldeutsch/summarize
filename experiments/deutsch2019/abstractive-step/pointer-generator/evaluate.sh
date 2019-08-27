expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [ "$#" -ne 2 ]; then
    echo "Usage: sh evaluate.sh <preprocessing-dataset> <use-context>"
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

output_dir=${expt_dir}/output/${preprocessing_dataset}/${context_dir}
results_dir=${expt_dir}/results/${preprocessing_dataset}/${context_dir}
mkdir -p ${results_dir}

for split in valid test; do
  python -m summarize.metrics.rouge \
    ${preprocess_dir}/${split}.jsonl.gz \
    ${output_dir}/${split}.jsonl \
    --gold-summary-field-name cloze \
    --model-summary-field-name cloze \
    --add-gold-wrapping-list \
    --add-model-wrapping-list \
    --compute-rouge-l \
    --silent \
    --output-file ${results_dir}/${split}.metrics.json
done
