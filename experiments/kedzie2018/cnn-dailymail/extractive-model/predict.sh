if [ "$#" -ne 2 ]; then
    echo "Usage: sh predict.sh <encoder> <extractor>"
    exit
fi

encoder=$1
extractor=$2

expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
model_file=${expt_dir}/model/${encoder}/${extractor}/model.tar.gz
output_dir=${expt_dir}/output/${encoder}/${extractor}

mkdir -p ${output_dir}

for split in valid test; do
  allennlp predict \
    --include-package summarize \
    --predictor sds-extractive-predictor \
    --output-file ${output_dir}/${split}.jsonl \
    --cuda-device 0 \
    --batch-size 16 \
    --silent \
    --use-dataset-reader \
    ${model_file} \
    https://s3.amazonaws.com/danieldeutsch/summarize/data/kedzie2018/cnn-dailymail/${split}.v1.0.jsonl.gz
done
