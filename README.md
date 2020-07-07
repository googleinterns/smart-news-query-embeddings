# Smart News Text Embedding

This is an open ended NLP project to derive embeddings from free form text.

This project uses the Apache license, as is Google's default.

# Installation

You will need Python 3.8 or higher to run this project. From the command line, run the following steps to install all dependencies after cloning this repo:

```
python3.8 -m venv bert_env
source bert_env/bin/activate
pip install -r requirements_keras.txt
```

# Running the Code

We use [this library](https://pypi.org/project/bert-for-tf2/) to instantiate a BERT layer that can be used in Keras models with TF 2.2. To learn how to run prediction on saved models or train new ones, run `python train_bert_keras_model --help` for more information on the exact configurations you can pass in. The script currently uses news article data which is available from the [New York Times API](https://developer.nytimes.com/).
