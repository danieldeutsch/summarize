expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

if [ "$#" -ne 2 ]; then
    echo "Usage: sh run.sh <preprocessing-dataset> <use-context>"
    exit
fi

sh ${expt_dir}/train.sh $1 $2
sh ${expt_dir}/predict.sh $1 $2
sh ${expt_dir}/evaluate.sh $1 $2
