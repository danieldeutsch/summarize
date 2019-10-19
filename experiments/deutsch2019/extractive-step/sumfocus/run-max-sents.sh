#!/bin/sh
#$ -cwd
if [ "$#" -ne 7 ]; then
    echo "Usage: sh run-max-sents.sh"
    echo "    <input-file> <output-file> <metrics-file> <beta> <topic-lambda> <context-lambda> <max-sents>"
    exit
fi

input_file=$1
output_file=$2
metrics_file=$3
beta=$4
topic_lambda=$5
context_lambda=$6
max_sents=$7

mkdir -p $(dirname ${output_file})
python -m summarize.models.cloze.sumfocus \
  ${input_file} \
  ${output_file} \
  ${beta} \
  ${topic_lambda} \
  ${context_lambda} \
  --max-sentences ${max_sents}

mkdir -p $(dirname ${metrics_file})
python -m summarize.metrics.rouge \
  ${input_file} \
  ${output_file} \
  --gold-summary-field-name cloze \
  --model-summary-field-name cloze \
  --add-gold-wrapping-list \
  --add-model-wrapping-list \
  --compute-rouge-l \
  --silent \
  --output-file ${metrics_file}
