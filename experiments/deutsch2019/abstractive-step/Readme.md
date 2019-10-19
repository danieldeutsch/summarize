# Abstractive Step
This directory contains the scripts to train the abstractive models on the preprocessed datasets.
First, the Pointer-Generator model needs to be trained using the `pointer-generator` directory.
Then, the model needs to be fine-tuned with the coverage loss using the `coverage` directory.

Each of the scripts requires two arguments, one which indicates which preprocessing dataset should be used (`lead`, `oracle`, or `extractive-model`), and a Boolean which indicates if the context should be used.
The coverage model will automatically reference the pretrained Pointer-Generator model.
The models will read the data from the corresponding directory under `extractive-step/<model>/preprocessed`, which should already exist before training.

## Saved Models
Here are links to the saved trained models, output, results, and preprocessed data.
<table>
  <thead>
    <tr>
      <th>Preprocessing</th>
      <th>Context</th>
      <th>Saved Model</th>
      <th>Model Output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=2 align="center">Oracle</td>
      <td align="center">True</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/model/oracle/context/model.tar.gz">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/model/oracle/context/model.tar.gz">Coverage</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/output/oracle/context/test.jsonl">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/output/oracle/context/test.jsonl">Coverage</a></td>
    </tr>
    <tr>
      <td align="center">False</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/model/oracle/no-context/model.tar.gz">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/model/oracle/no-context/model.tar.gz">Coverage</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/output/oracle/no-context/test.jsonl">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/output/oracle/no-context/test.jsonl">Coverage</a></td>
    </tr>
    <tr>
      <td rowspan=2 align="center">Lead</td>
      <td align="center">True</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/model/lead/context/model.tar.gz">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/model/lead/context/model.tar.gz">Coverage</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/output/lead/context/test.jsonl">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/output/lead/context/test.jsonl">Coverage</a></td>
    </tr>
    <tr>
      <td align="center">False</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/model/lead/no-context/model.tar.gz">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/model/lead/no-context/model.tar.gz">Coverage</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/output/lead/no-context/test.jsonl">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/output/lead/no-context/test.jsonl">Coverage</a></td>
    </tr>
    <tr>
      <td rowspan=2 align="center">Extractive Model</td>
      <td align="center">True</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/model/extractive-model/context/model.tar.gz">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/model/extractive-model/context/model.tar.gz">Coverage</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/output/extractive-model/context/test.jsonl">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/output/extractive-model/context/test.jsonl">Coverage</a></td>
    </tr>
    <tr>
      <td align="center">False</td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/model/extractive-model/no-context/model.tar.gz">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/model/extractive-model/no-context/model.tar.gz">Coverage</a></td>
      <td align="center"><a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/pointer-generator/output/extractive-model/no-context/test.jsonl">Pointer-Generator</a>, <a href="https://danieldeutsch.s3.amazonaws.com/summarize/experiments/deutsch2019/v1.1/abstractive-step/coverage/output/extractive-model/no-context/test.jsonl">Coverage</a></td>
    </tr>
  </tbody>
</table>
