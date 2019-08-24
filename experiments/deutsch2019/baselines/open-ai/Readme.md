# OpenAI Language Model
The OpenAI Language Model ([Radford et al., 2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)) serves as a baseline for the summary cloze task.
The language model conditions on the context of the summary and generates the next sentence.
The cited document is not used at all.
The purpose of the experiment is to measure how well a system can do without access to the reference text.

## Setup
Before using the OpenAI language model, you first need to download the model
```
sh experiments/deutsch2019/baselines/open-ai/setup.sh
```
For more documentation on the model and its parameters, see the official [Github repository](https://github.com/openai/gpt-2).
