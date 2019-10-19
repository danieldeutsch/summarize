# Extractive Model
This directory contains the scripts to train the extractive model (referred to as `ContentSelector` in the paper).
Each of the `train.sh`, `predict.sh`, `evaluate.sh`, and `preprocess.sh` scripts require two Boolean arguments.
The first indicates whether or not the topic should be used, and the second indicates whether or not the context should be used.

The `train.sh` script will train the model.
`predict.sh` and `evaluate.sh` will run inference and evaluate both the extractive model (select the best one sentence) and the extractive preprocessing step (take as many sentences until a budget is met).
Then, the `preprocess.sh` script will use the trained model to preprocess the input documents and reduce the input length for the abstractive step.

## Saved Models
Here are links to the saved trained models, output, results, and preprocessed data.
<table>
  <thead>
    <tr>
      <th>Topics</th>
      <th>Context</th>
      <th>Saved Model</th>
      <th>Model Output</th>
      <th>Preprocessed Data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">True</td>
      <td align="center">True</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/model/topics/context/model.tar.gz">Link</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/topics/context/test.max-tokens.jsonl">200 Tokens</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/topics/context/test.max-sents.jsonl">1 Sentence</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/preprocessed/topics/context/train.jsonl.gz">Train</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/preprocessed/topics/context/valid.jsonl.gz">Valid</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/preprocessed/topics/context/test.jsonl.gz">Test</a></td>
    </tr>
    <tr>
      <td align="center">True</td>
      <td align="center">False</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/model/topics/no-context/model.tar.gz">Link</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/topics/no-context/test.max-tokens.jsonl">200 Tokens</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/topics/no-context/test.max-sents.jsonl">1 Sentence</a></td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">False</td>
      <td align="center">True</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/model/no-topics/context/model.tar.gz">Link</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/no-topics/context/test.max-tokens.jsonl">200 Tokens</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/no-topics/context/test.max-sents.jsonl">1 Sentence</a></td>
      <td align="center">-</td>
    </tr>
    <tr>
      <td align="center">False</td>
      <td align="center">False</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/model/no-topics/no-context/model.tar.gz">Link</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/no-topics/no-context/test.max-tokens.jsonl">200 Tokens</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/extractive-step/extractive-model/output/no-topics/no-context/test.max-sents.jsonl">1 Sentence</a></td>
      <td align="center">-</td>
    </tr>
  </tbody>
</table>
