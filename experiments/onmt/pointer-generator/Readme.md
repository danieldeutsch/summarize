## Summarize Commands
To train, predict, and evaluate with the Summarize models, run
```
sh experiments/onmt/pointer-generator/train.sh
sh experiments/onmt/pointer-generator/predict.sh
sh experiments/onmt/pointer-generator/evaluate.sh
```
The output and metrics will be written to the `output` and `results` directories, respectively.
The trained model can be downloaded here TODO

## OpenNMT Commands
To train the OpenNMT model, run the following command from the root of the `OpenNMT-py` directory:
```
mkdir -p models
python train.py \
  -save_model models/pointer-generator \
  -data data/cnndm/CNNDM \
  -copy_attn \
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
  -reuse_copy_attn \
  -copy_loss_by_seqlength \
  -bridge \
  -seed 777 \
  -world_size 1 \
  -gpu_ranks 0
```
To evaluate the model with the minimum-length constraint, run this command:
```
mkdir -p output/pointer-generator

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/pointer-generator_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/pointer-generator/test.none.out

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/pointer-generator_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/pointer-generator/test.min-length.out \
  -min_length 35

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/pointer-generator_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/pointer-generator/test.repeated-trigrams.out \
  -min_length 35 \
  -block_ngram_repeat 3 \
  -ignore_when_blocking "." "</t>" "<t>"

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/pointer-generator_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/pointer-generator/test.length.out \
  -min_length 35 \
  -block_ngram_repeat 3 \
  -ignore_when_blocking "." "</t>" "<t>" \
  -length_penalty wu \
  -alpha 0.9

python translate.py \
  -gpu 0 \
  -batch_size 20 \
  -beam_size 10 \
  -model models/pointer-generator_step_200000.pt \
  -src data/cnndm/test.txt.src \
  -output output/pointer-generator/test.coverage.out \
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
  TODO
```
Finally, compute ROUGE using the following commands:
```
python -m summarize.metrics.rouge \
  ... TODO
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
      <td>min-length</td>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>d</td>
      <td>e</td>
      <td>f</td>
    </tr>
    <tr>
      <td>min-length, repeated ngrams</td>
      <td>a</td>
      <td>b</td>
      <td>c</td>
      <td>d</td>
      <td>e</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
