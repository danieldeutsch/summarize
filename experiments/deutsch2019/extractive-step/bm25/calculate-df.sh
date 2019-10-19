expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir="${expt_dir}/output"
mkdir -p ${output_dir}

python -m summarize.models.cloze.bm25.calculate_df \
  https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/train.v1.1.jsonl.gz \
  ${output_dir}/df.jsonl.gz
