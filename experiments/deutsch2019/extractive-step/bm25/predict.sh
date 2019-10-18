expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
output_dir="${expt_dir}/output"
mkdir -p ${output_dir}

max_words=200
max_sents=1

for split in valid test; do
  python -m summarize.models.cloze.bm25.bm25 \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${output_dir}/df.jsonl.gz \
    ${output_dir}/${split}.max-words.jsonl \
    --max-words ${max_words}

  python -m summarize.models.cloze.bm25.bm25 \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${output_dir}/df.jsonl.gz \
    ${output_dir}/${split}.max-sents.jsonl \
    --max-sentences ${max_sents} \
    --flatten
done
