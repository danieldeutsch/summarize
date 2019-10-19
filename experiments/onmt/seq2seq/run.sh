expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

sh ${expt_dir}/train.sh
sh ${expt_dir}/replace-config.sh
sh ${expt_dir}/predict.sh
sh ${expt_dir}/evaluate.sh
