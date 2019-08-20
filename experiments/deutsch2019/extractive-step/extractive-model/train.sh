if [ "$#" -ne 2 ]; then
    echo "Usage: sh train.sh <use-topics> <use-context>"
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
model_dir=${expt_dir}/model/${topics_dir}/${context_dir}
model_config=${expt_dir}/model.jsonnet

if [ -d ${model_dir} ]; then
  read -p "remove directory ${model_dir}? [y/n] " yn
  case $yn in
        [Yy]* ) rm -rf ${model_dir};;
        [Nn]* ) ;;
        * ) echo "Please answer yes or no.";;
  esac
fi

export USE_TOPICS=${use_topics}
export USE_CONTEXT=${use_context}
allennlp train \
  --include-package summarize \
  --serialization-dir ${model_dir} \
  ${model_config}
