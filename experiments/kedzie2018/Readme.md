# Kedzie 2018
This is a partial reimplementation of [Content Selection in Deep Learning Models of Summarization](https://arxiv.org/abs/1810.12343) by Kedzie et al. (2018).

## Instructions
First, prepare the necessary data under `data/kedzie2018`.
Then, each directory of the experiment corresponds to a different dataset and model with its own script to train, predict, and evaluate.

## Results
Below are the reproduction results for the CNN/DailyMail dataset.
<table>
  <thead>
    <tr>
      <th rowspan=2>Extractor</th>
      <th rowspan=2>Encoder</th>
      <th colspan=2>R2-Recall</th>
      <th rowspan=2>Saved Model</th>
    </tr>
    <tr>
      <th>Reported</th>
      <th>Reproduced</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Lead</th>
      <th>-</th>
      <td align="center">24.4</td>
      <td align="center">24.4</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <th rowspan=3>RNN</th>
      <th>Avg</th>
      <td align="center">25.4</td>
      <td align="center">25.5</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/kedzie2018/cnn-dailymail/extractive-model/model/avg/rnn/model.tar.gz">Link</a></td>
    </tr>
    <tr>
      <th>RNN</th>
      <td align="center">25.4</td>
      <td align="center">25.4</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/kedzie2018/cnn-dailymail/extractive-model/model/rnn/rnn/model.tar.gz">Link</a></td>
    </tr>
    <tr>
      <th>CNN</th>
      <td align="center">25.1</td>
      <td align="center">-</td>
      <td align="center">-</td>
    </tr>
    <tr>
      <th>Oracle</th>
      <th>-</th>
      <td align="center">36.2</td>
      <td align="center">37.3</td>
      <td align="center">-</td>
    </tr>
  </tbody>
</table>
