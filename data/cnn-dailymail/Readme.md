# CNN/DailyMail
## Setup
To setup the CNN/DailyMail dataset, first run the following command, which will download the CNN and DailyMail stories from [here](https://cs.nyu.edu/~kcho/DMQA/) and extract the documents and summaries:
```
python -m summarize.data.dataset_setup.cnn_dailymail data/cnn-dailymail
```
The script is largely based on the one from See et al. (2017), modified to skip the tokenization and normalization.
(This is in case some model requires the original text.)

There are 287,113 training, 13,368 validation, and 11,490 testing instances.
The number of training instances is different from the reported number in [See et al. (2017)](https://arxiv.org/pdf/1704.04368.pdf) (287,226), likely because of those note from the [official paper dataset repository](https://github.com/abisee/cnn-dailymail):

> Warning: These files contain a few (114, in a dataset of over 300,000) examples for which the article text is missing - see for example cnn/stories/72aba2f58178f2d19d3fae89d5f3e9a4686bc4bb.story.
> The Tensorflow code has been updated to discard these examples.

(although the 114 number is off by 1).
The number does match later work, such as [Kedzie et al. (2018)](https://arxiv.org/pdf/1810.12343.pdf).

The files can be downloaded from AWS here:

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
        <td>CNN</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn/train.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn/valid.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn/test.v1.1.jsonl.gz</td>
      </tr>
      <tr>
        <td>DailyMail</d>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/dailymail/train.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/dailymail/valid.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/dailymail/test.v1.1.jsonl.gz</td>
      </tr>
      <tr>
        <td>CNN/DailyMail</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/train.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/valid.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/test.v1.1.jsonl.gz</td>
      </tr>
    </tbody>
  </table>

### Tokenization
If you want to preprocess the dataset with tokenization, run the following tokenization command on each of the relevant splits:
```
python -m summarize.data.dataset_setup.tokenize \
  data/cnn-dailymail/cnn-dailymail/train.jsonl.gz \
  data/cnn-dailymail/cnn-dailymail/train.tokenized.jsonl.gz \
  document summary \
  --backend nltk
```
We chose nltk as the backend because it handled dataset-specific peculiarities.
For example, many of the documents begin with sentences like
> Marseille, France (CNN)The French ...

Spacy tokenized the sentence as
> Marseille , France ( CNN)The French ...

nltk tokenized it correctly:
> Marseille , France ( CNN ) The French ...

See et al. (2017) does tokenization with Stanford CoreNLP, which can be downloaded [here](https://github.com/JafferWilson/Process-Data-of-CNN-DailyMail) if necessary.

The tokenized files can be downloaded from AWS here:

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
        <td>CNN</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn/train.tokenized.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn/valid.tokenized.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn/test.tokenized.v1.1.jsonl.gz</td>
      </tr>
      <tr>
        <td>DailyMail</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/dailymail/train.tokenized.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/dailymail/valid.tokenized.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/dailymail/test.tokenized.v1.1.jsonl.gz</td>
      </tr>
      <tr>
        <td>CNN/DailyMail</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/train.tokenized.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/valid.tokenized.v1.1.jsonl.gz</td>
        <td>https://s3.amazonaws.com/danieldeutsch/summarize/data/cnn-dailymail/cnn-dailymail/test.tokenized.v1.1.jsonl.gz</td>
      </tr>
    </tbody>
  </table>


## Changelog
### v1.1
- Added further sentence tokenization to the lines from the CNN and DailyMail articles as I observed that a significant number of lines had multiple sentences.
This only impacts sentence extractive models since abstractive models concatenate the document together during preprocessing.
