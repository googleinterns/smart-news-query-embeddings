# Smart News Text Embedding

This is an open ended NLP project to derive embeddings from free form text.

This project uses the Apache license, as is Google's default.

# Running the Code

To run saved models or train new ones, you will need Python 3.7. From the command line, run `./install.sh` to create a new virtual environment with all necessary dependencies. Then, after activating the environment, run:

`python test_bert_trainer.py -h`

for more information on the exact configurations you can pass in. The script currently uses news article data which is available from the [New York Times API](https://developer.nytimes.com/).
