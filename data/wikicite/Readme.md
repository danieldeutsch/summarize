# WikiCite
The WikiCite dataset is a collection of summary cloze instances collected from Wikipedia.
For more details, please see https://github.com/danieldeutsch/wikicite.

## Setup
The `setup.sh` script downloads the original dataset and tokenizes the text fields.
The original dataset and tokenized versions can be downloaded here:

<table>
  <thead>
    <tr>
      <th>Corpus</th>
      <th>Train</th>
      <th>Valid</th>
      <th>Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Original</td>
      <td>https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/train.v1.1.jsonl.gz</td>
      <td>https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/valid.v1.1.jsonl.gz</td>
      <td>https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/test.v1.1.jsonl.gz</td>
    </tr>
    <tr>
      <td>Tokenized</td>
      <td>https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/train.tokenized.v1.1.jsonl.gz</td>
      <td>https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/valid.tokenized.v1.1.jsonl.gz</td>
      <td>https://danieldeutsch.s3.amazonaws.com/summarize/data/wikicite/test.tokenized.v1.1.jsonl.gz</td>
    </tr>
  </tbody>
</table>


## Citation
If you use this dataset, please cite the following paper:
```
@inproceedings{DeutschRo19,
    author = {Daniel Deutsch and Dan Roth},
    title = {{Summary Cloze: A New Task for Content Selection in Topic-Focused Summarization}},
    booktitle = {Proc. of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2019},
    url = "https://cogcomp.seas.upenn.edu/papers/DeutschRo19.pdf",
    funding = {ARL},
}
```
