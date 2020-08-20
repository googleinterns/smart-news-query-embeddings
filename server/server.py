"""
Copyright 2020 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
A small server that can be run on Google Cloud Platform to expose a REST
endpoint for getting embeddings and nearest neighbors of input articles.
"""

import tornado
import os
import json
from tornado.web import RequestHandler, Application
import numpy as np
import pandas as pd
import tensorflow as tf
from smart_news_query_embeddings.preprocessing.bert_tokenizer import \
create_tokenizer, tokenize_data

# We first set up the tokenizer and load in the trained model that we want to serve.
home = os.path.expanduser('~')
bert_dir = os.path.join(home, 'smart-news-query-embeddings/uncased_L-12_H-768_A-12/')
tokenizer = create_tokenizer(bert_dir)
exp_name = 'baseline_model'
exp_dir = os.path.join(home, 'smart-news-query-embeddings/experiments', exp_name, 'model')
keras_model = tf.keras.models.load_model(exp_dir)

# We then load in the embeddings generated from the trained model,
# which we will use to determine the nearest neighbors of input sentences.
embeddings_path = os.path.join(home, 'smart-news-query-embeddings/experiments', exp_name, 'embeddings')
train_embeddings_path = os.path.join(embeddings_path, 'train_embeddings.npy')
valid_embeddings_path = os.path.join(embeddings_path, 'valid_embeddings.npy')
train_embeddings = np.load(train_embeddings_path)
valid_embeddings = np.load(valid_embeddings_path)
all_embeddings = np.concatenate((train_embeddings, valid_embeddings))

# We make all the embeddings have unit norm to make the nearest neighbor calculation simpler.
norms = np.linalg.norm(all_embeddings, axis=1)
all_embeddings /= norms[:, np.newaxis]

# We also load in all the sentences used in the data to train the model, so that we can
# return actual sentences in the response corresponding to the indices of the nearest neighbors
# in the embedding space.
data_path = os.path.join(home, 'smart-news-query-embeddings/experiments', exp_name, 'data')
train_sentences_path = os.path.join(data_path, 'train_sentences.pkl')
valid_sentences_path = os.path.join(data_path, 'valid_sentences.pkl')
train_sentences = pd.read_pickle(train_sentences_path)
valid_sentences = pd.read_pickle(valid_sentences_path)
sentences = pd.concat((train_sentences, valid_sentences))

class PredictHandler(RequestHandler):    

    def get(self):

        """
        The main request handler for this server. Upon receiving a request with an input article,
        this function does the following:

        1) Tokenize and pad the input article into the correct shape for the BERT model.
        2) Runs a partial forward pass of the loaded model to get the embedding of the sequence.
        3) Computes the dot product of the embedding with all embeddings in the dataset used
        to train the model.
        4) Takes the indices of the top 10 dot product values (similarity scores), and returns
        the sentences at those indices along with the embedding vector of the article.
        """

        article = self.get_argument('article', None, True)
        input_seq, _ = tokenize_data([article], [0], tokenizer, 128, 64)
        embedding = keras_model.get_embedding(input_seq).numpy()[0].astype(float)
        embedding /= np.linalg.norm(embedding)
        similarities = all_embeddings.dot(embedding)
        top_indices = np.argsort(similarities)[-10:]
        top_sentences = list(sentences.iloc[top_indices])
        resp = {
            'embedding': list(embedding),
            'similar_sentences': top_sentences,
        }
        self.write(json.dumps(resp, indent=4))

    def check_origin(self, origin):
        return True

if __name__ == '__main__':
    application = Application([
        (r"/predict", PredictHandler),
    ])
    application.listen(8888)
    print('STARTING APPLICATION ON LOCALHOST:8888')
    tornado.ioloop.IOLoop.current().start()
