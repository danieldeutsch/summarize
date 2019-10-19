expt_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
preprocess_dir="${expt_dir}/preprocessed"
mkdir -p ${preprocess_dir}

for split in train valid test; do
  temp_file=$(mktemp)
  python -m summarize.models.cloze.lead \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${temp_file} \
    --max-tokens 200 \
    --field-name document \
    --keep-sentences

  python -m summarize.utils.copy_jsonl_fields \
    ${temp_file} \
    https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/${split}.v1.1.jsonl.gz \
    ${preprocess_dir}/${split}.jsonl.gz \
    --field-names document document

  rm ${temp_file}
done
