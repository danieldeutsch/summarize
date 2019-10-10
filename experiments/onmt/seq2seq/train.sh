expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
model_dir=${expt_dir}/model
model_config=${expt_dir}/model.jsonnet

if [ -d ${model_dir} ]; then
  read -p "remove directory ${model_dir}? [y/n] " yn
  case $yn in
        [Yy]* ) rm -rf ${model_dir};;
        [Nn]* ) ;;
        * ) echo "Please answer yes or no.";;
  esac
fi

allennlp train \
  --include-package summarize \
  --serialization-dir ${model_dir} \
  --overrides '{"model.run_beam_search": false}' \
  ${model_config}
