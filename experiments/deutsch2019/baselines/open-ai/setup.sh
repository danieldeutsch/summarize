cwd=$(pwd)
expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

pushd ${expt_dir}
python ${cwd}/external/gpt-2/download_model.py 345M
