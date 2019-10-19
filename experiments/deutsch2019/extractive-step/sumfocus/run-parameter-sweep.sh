expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
max_words_output_dir="${expt_dir}/sweep/max-words"
max_sents_output_dir="${expt_dir}/sweep/max-sents"
log_dir="${expt_dir}/logs"
mkdir -p ${max_words_output_dir} ${max_sents_output_dir} ${log_dir}

max_words=200
max_num_sents=1

for beta in 0.1 0.5 1.0 2.0; do
  for topic_lambda in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for context_lambda in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
      for split in valid; do
        name="beta_${beta}.topic-lambda-${topic_lambda}.context-lambda-${context_lambda}"
        gold_file="https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz"
        model_file="${max_words_output_dir}/${split}.${name}.jsonl"
        metrics_file="${max_words_output_dir}/${split}.${name}.metrics.json"

        stdout=${log_dir}/${name}-words.stdout
        stderr=${log_dir}/${name}-words.stderr
        qsub -N ${name} -o ${stdout} -e ${stderr} \
          ${expt_dir}/run-max-words.sh ${gold_file} ${model_file} ${metrics_file} ${beta} ${topic_lambda} ${context_lambda} ${max_words}

        model_file="${max_sents_output_dir}/${split}.${name}.jsonl"
        metrics_file="${max_sents_output_dir}/${split}.${name}.metrics.json"

        stdout=${log_dir}/${name}-sents.stdout
        stderr=${log_dir}/${name}-sents.stderr
        qsub -N ${name} -o ${stdout} -e ${stderr} \
          ${expt_dir}/run-max-sents.sh ${gold_file} ${model_file} ${metrics_file} ${beta} ${topic_lambda} ${context_lambda} ${max_num_sents}
      done
    done
  done
done
