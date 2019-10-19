# Deutsch 2019
This directory contains the experiments related to "Summary Cloze: A New Task for Content Selection in Topic-Focused Summarization" by Deutsch and Roth (2019).

## Demo
A demo of the final models (with the topics and context) can be viewed by clicking this badge:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/danieldeutsch/summarize/21054f43de1b363aba1e1283d62736e5117877bf?filepath=experiments%2Fdeutsch2019%2Fdemo.ipynb)

If you run the Jupyter Notebook on the MyBinder servers, the abstractive model takes around 30 to 60 seconds to produce the output.

## Instructions
First, it may be necessary to checkout [this commit](https://github.com/danieldeutsch/summarize/releases/tag/emnlp2019) since there could have been breaking changes to the code since the original models were trained.

Then, setup the WikiCite dataset by running the setup script in `data/deutsch2019`.

Each of the directories contains the scripts to run the different models from the paper.
The `baselines` directory contains code for some baseline models, such as the lead, oracle, and language model baselines.
The `extractive-step` directory contains the code for the extractive models and extractive preprocessing steps.
The `abstractive-step` directory contains the code for training the abstractive models, both the base Pointer-Generator model and the fine-tuned model with the coverage loss.
The directories contain documentation with extra information, results, and saved models.

If you use any of the code or data from this experiment, please cite the following paper:
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
