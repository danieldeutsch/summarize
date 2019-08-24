#!/bin/sh
#$ -cwd
if [ "$#" -ne 7 ]; then
    echo "Usage: sh run-max-words.sh"
    echo "    <input-file> <output-file> <metrics-file> <beta> <topic-lambda> <context-lambda> <max-words>"
    exit
fi

input_file=$1
output_file=$2
metrics_file=$3
beta=$4
topic_lambda=$5
context_lambda=$6
max_words=$7

mkdir -p $(dirname ${output_file})
python -m summarize.models.cloze.sumfocus \
  ${input_file} \
  ${output_file} \
  ${beta} \
  ${topic_lambda} \
  ${context_lambda} \
  --max-words ${max_words}

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
  --max-words 200 \
  --output-file ${metrics_file}
