## Summarize Commands
To train, predict, and evaluate with the Summarize models, run
```
sh experiments/onmt/pointer-generator/train.sh
sh experiments/onmt/pointer-generator/replace-config.sh
sh experiments/onmt/pointer-generator/predict.sh
sh experiments/onmt/pointer-generator/evaluate.sh
```
The output and metrics will be written to the `output` and `results` directories, respectively.
The trained model can be downloaded [here](https://danieldeutsch.s3.amazonaws.com/summarize/experiments/onmt/v1.0/pointer-generator/model/model.tar.gz).

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
  <OpenNMT-py>/data/cnndm/test.txt.tgt.tagged \
  <OpenNMT-py>/data/cnndm/test.txt.tgt.tagged.jsonl

  python experiments/onmt/convert_to_jsonl.py \
    <OpenNMT-py>/output/pointer-generator/test.repeated-trigrams.out \
    <OpenNMT-py>/output/pointer-generator/test.repeated-trigrams.jsonl
```
Finally, compute ROUGE using the following commands:
```
python -m summarize.metrics.rouge \
  data/cnndm/test.txt.tgt.tagged.jsonl \
  output/pointer-generator/test.min-length.jsonl \
  --compute-rouge-l
```

## Results
I do not know what the cause of the differences between the ROUGE scores is.
I have done experiments where I verify the two libraries compute the same loss and return the same predictions from inference, but for some reason end up with different scores after training.
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
      <td>34.19</td>
      <td>14.35</td>
      <td>30.22</td>
      <td>34.50</td>
      <td>14.79</td>
      <td>25.05</td>
    </tr>
    <tr>
      <td>+disallow repeated trigrams</td>
      <td>36.18</td>
      <td>15.47</td>
      <td>31.42</td>
      <td>36.58</td>
      <td>15.98</td>
      <td>25.82</td>
    </tr>
    <tr>
      <td>+length penalty</td>
      <td>39.15</td>
      <td>16.69</td>
      <td>33.07</td>
      <td>39.53</td>
      <td>17.11</td>
      <td>26.57</td>
    </tr>
    <tr>
      <td>+coverage penalty</td>
      <td>37.82</td>
      <td>15.72</td>
      <td>32.14</td>
      <td>38.34</td>
      <td>16.27</td>
      <td>26.03</td>
    </tr>
  </tbody>
</table>
