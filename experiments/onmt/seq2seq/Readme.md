## Summarize Commands
To train, predict, and evaluate with the Summarize models, run
```
sh experiments/onmt/seq2seq/train.sh
sh experiments/onmt/seq2seq/replace-config.sh
sh experiments/onmt/seq2seq/predict.sh
sh experiments/onmt/seq2seq/evaluate.sh
```
Because the fully-constrained inference is very slow, we use the `overrides` parameter during training to remove some of the constraints to make it much faster.
As a consequence, we have to replace the config file in the trained model with the original one to use the fully-constrained inference by default with the `replace-config.sh` script.

The output and metrics will be written to the `output` and `results` directories, respectively.
The trained model can be downloaded [here](https://danieldeutsch.s3.amazonaws.com/summarize/experiments/onmt/v1.0/seq2seq/model/model.tar.gz).

## OpenNMT Commands
To train the OpenNMT model, run the following command from the root of the `OpenNMT-py` directory:
```
mkdir -p models
python train.py \
  -save_model models/seq2seq \
  -data data/cnndm/CNNDM \
  -global_attention mlp \
  -word_vec_size 128 \
  -rnn_size 512 \
  -layers 1 \
  -encoder_type brnn \
  -train_steps 200000 \
  -max_grad_norm 2 \
  -dropout 0. \
  -batch_size 16 \
  -valid_batch_size 16 \
  -optim adagrad \
  -learning_rate 0.15 \
  -adagrad_accumulator_init 0.1 \
  -bridge \
  -seed 777 \
  -world_size 1 \
  -gpu_ranks 0
```
To evaluate the model with the various constraints, run this command:
```
mkdir -p output/seq2seq

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/seq2seq_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/seq2seq/test.min-length.out \
  -min_length 35

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/seq2seq_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/seq2seq/test.repeated-trigrams.out \
  -min_length 35 \
  -block_ngram_repeat 3 \
  -ignore_when_blocking "." "</t>" "<t>"

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/seq2seq_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/seq2seq/test.length.out \
  -min_length 35 \
  -block_ngram_repeat 3 \
  -ignore_when_blocking "." "</t>" "<t>" \
  -length_penalty wu \
  -alpha 0.9

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/seq2seq_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/seq2seq/test.coverage.out \
  -min_length 35 \
  -block_ngram_repeat 3 \
  -ignore_when_blocking "." "</t>" "<t>" \
  -stepwise_penalty \
  -coverage_penalty summary \
  -beta 5 \
  -length_penalty wu \
  -alpha 0.9
```

In order to evaluate the output, both the ground-truth and model predictions need to be converted to the jsonl format using these commands, which should be run from the root of the `summarize` library:
```
python experiments/onmt/convert_to_jsonl.py \
  <OpenNMT-py>/data/cnndm/test.txt.tgt.tagged \
  <OpenNMT-py>/data/cnndm/test.txt.tgt.tagged.jsonl

python experiments/onmt/convert_to_jsonl.py \
  <OpenNMT-py>/output/seq2seq/test.repeated-trigrams.out \
  <OpenNMT-py>/output/seq2seq/test.repeated-trigrams.jsonl
```
Finally, compute ROUGE using the following commands:
```
python -m summarize.metrics.rouge \
  data/cnndm/test.txt.tgt.tagged.jsonl \
  output/seq2seq/test.min-length.jsonl \
  --compute-rouge-l
```

## Results
<table>
  <thead>
    <tr>
      <th rowspan=2>Constraints</th>
      <th colspan=3>Summarize</th>
      <th colspan=3>OpenNMT</th>
    </tr>
    <tr>
      <th>R1-F1</th>
      <th>R2-F1</th>
      <th>RL-F1</th>
      <th>R1-F1</th>
      <th>R2-F1</th>
      <th>RL-F1</th>
    </tr>
  </thead>
  <tbody>
  <tr>
    <td>+min-length</td>
    <td>28.66</td>
    <td>10.70</td>
    <td>26.01</td>
    <td>27.80</td>
    <td>10.10</td>
    <td>21.05</td>
  </tr>
  <tr>
    <td>+disallow repeated trigrams</td>
    <td>32.92</td>
    <td>12.90</td>
    <td>28.68</td>
    <td>32.26</td>
    <td>12.30</td>
    <td>22.58</td>
  </tr>
  <tr>
    <td>+length penalty</td>
    <td>34.12</td>
    <td>13.32</td>
    <td>29.44</td>
    <td>33.26</td>
    <td>12.71</td>
    <td>22.96</td>
  </tr>
  <tr>
    <td>+coverage penalty</td>
    <td>34.09</td>
    <td>13.26</td>
    <td>29.40</td>
    <td>33.23</td>
    <td>12.60</td>
    <td>22.90</td>
  </tr>
  </tbody>
</table>
