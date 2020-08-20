# Smart News Text Embedding

This is an open ended NLP project to derive embeddings from free form text.

This project uses the Apache license, as is Google's default.

# Installation

You will need Python 3.8 or higher to run this project. From the command line, run the following steps to install all dependencies after cloning this repo:

```
python3.8 -m venv bert_env
source bert_env/bin/activate
python setup.py install
```

# Running the Code

We use [this library](https://pypi.org/project/bert-for-tf2/) to instantiate a BERT layer that can be used in Keras models with TF 2.2. To learn how to run prediction on saved models or train new ones, run `python training/train_model.py --help` for more information on the exact configurations you can pass in. The script currently uses news article data which is available from the [New York Times API](https://developer.nytimes.com/).

# Directory Structure

Below is a birds-eye view of the directory structure of this project:

```
config/
  requirements_keras.txt
  requirements_keras_gcp.txt
data/
  get_nyt_articles.py
smart_news_query_embeddings/
  __init__.py
  models/
    __init__.py
    bert_keras_model.py
    two_tower_model.py
  preprocessing/
    __init__.py
    bert_tokenizer.py
    specificity_scores.py
  trainers/
    __init__.py
    bert_model_trainer.py
    bert_model_specificity_score_trainer.py
    two_tower_model_trainer.py
  tests/
    [all unit tests here]
training/
  train_model.py
```
