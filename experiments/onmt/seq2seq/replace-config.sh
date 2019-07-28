expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
model_config=${expt_dir}/model.jsonnet
model_tar=${expt_dir}/model/model.tar.gz

python -m summarize.utils.replace_config \
  ${model_tar} \
  ${model_tar} \
  ${model_config}
