expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [ "$#" -ne 2 ]; then
    echo "Usage: sh train.sh <preprocessing-dataset> <use-context>"
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
model_config=${expt_dir}/model.jsonnet

if [ -d ${model_dir} ]; then
  read -p "remove directory ${model_dir}? [y/n] " yn
  case $yn in
        [Yy]* ) rm -rf ${model_dir};;
        [Nn]* ) ;;
        * ) echo "Please answer yes or no.";;
  esac
fi

export DATA_DIR=${preprocess_dir}
export USE_CONTEXT=${use_context}
allennlp train \
  --include-package summarize \
  --serialization-dir ${model_dir} \
  ${model_config}
