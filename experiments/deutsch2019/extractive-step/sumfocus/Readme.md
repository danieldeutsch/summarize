# SumFocus
An implementation of SumFocus from [Vanderwende et al. (2007)](https://www.cis.upenn.edu/~nenkova/papers/ipm.pdf).
`run-parameter-sweep.sh` will run a parameter sweep to find the best settings of the unigram probability distribution smoothing parameter (`beta` in the code) and the interpolation parameters between the document, topic, and context (`topic_lambda` and `context_lambda` in the code) using the NLP Grid for parallelization.
To analyze the results, run the python script `analyze_results.py` which will output what the best hyperparameter settings were for all variations of using and not using the topic and context.

After the best hyperparameter settings are found, you can run the model on the test data to compute Rouge by running
```
sh experiments/deutsch2019/extractive-step/sumfocus/run-max-words.sh \
  https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/test.v1.1.jsonl.gz \
  experiments/deutsch2019/extractive-step/sumfocus/output/test.max-words.jsonl \
  experiments/deutsch2019/extractive-step/sumfocus/output/test.max-words.metrics.jsonl \
  <beta> \
  <topic_lambda> \
  <context_lambda> \
  200

sh experiments/deutsch2019/extractive-step/sumfocus/run-max-sents.sh \
  https://danieldeutsch.s3.amazonaws.com/summarize/data/deutsch2019/test.v1.1.jsonl.gz \
  experiments/deutsch2019/extractive-step/sumfocus/output/test.max-sents.jsonl \
  experiments/deutsch2019/extractive-step/sumfocus/output/test.max-sents.metrics.jsonl \
  <beta> \
  <topic_lambda> \
  <context_lambda> \
  1
```
