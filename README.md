# chuchikaestli

Find the full description of the hands-on task here: https://tiny.cc/versteisch-bahnhof


chuchikaestli is a [Swiss German dialect](https://en.wikipedia.org/wiki/Swiss_German) predictor using TF-IDF vector 
representations and a Random Forest classifier. 

The evaluation is based on a publicly available Swiss German 
[kaggle competition](https://www.kaggle.com/c/swiss-dialect-identification/overview). This dataset is based on four
different dialects:

```
BE Bernese
LU Lucerne
ZH Zurich
BS Basel
```

Whereby the training set consists of 15573 example sentences, wheres as the test set consists of 2499 example sentences.

## Requirements

Python3 is required.

First, install `pipenv` using `pip`:

```bash
pip install --user pipenv
```

## Installation

To load all dependencies into an own virtual environment:

```bash
pipenv install
```

Next, you can import the created virtual environment into your preferred IDE and activate it in your shell:

```bash
pipenv shell
```

## Usage

You can train the model either by `train_dialect` (fixed parameter setting) or `train_dialect_hyperparameter` (grid 
search over different parameter settings). In both cases, the best parameters are logged to the console.