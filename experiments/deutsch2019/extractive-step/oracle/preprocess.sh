expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
preprocess_dir="${expt_dir}/preprocessed"
mkdir -p ${preprocess_dir}

for split in train valid test; do
  temp_file=$(mktemp)
  python -m summarize.utils.extract_cloze_from_labels \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.0.jsonl.gz \
    ${temp_file} \
    --field-name document \
    --keep-sentences

  python -m summarize.utils.copy_jsonl_fields \
    ${temp_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.0.jsonl.gz \
    ${preprocess_dir}/${split}.jsonl.gz \
    document

  rm ${temp_file}
done
