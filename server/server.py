import tornado
import os
import json
from tornado.web import RequestHandler, Application
import numpy as np
import pandas as pd
import tensorflow as tf
from smart_news_query_embeddings.preprocessing.bert_tokenizer import \
create_tokenizer, tokenize_data

home = os.path.expanduser('~')
bert_dir = os.path.join(home, 'smart-news-query-embeddings/uncased_L-12_H-768_A-12/')
tokenizer = create_tokenizer(bert_dir)
exp_name = 'baseline_model'
exp_dir = os.path.join(home, 'smart-news-query-embeddings/experiments', exp_name, 'model')
keras_model = tf.keras.models.load_model(exp_dir)
embeddings_path = os.path.join(home, 'smart-news-query-embeddings/experiments', exp_name, 'embeddings')
train_embeddings_path = os.path.join(embeddings_path, 'train_embeddings.npy')
valid_embeddings_path = os.path.join(embeddings_path, 'valid_embeddings.npy')
train_embeddings = np.load(train_embeddings_path)
valid_embeddings = np.load(valid_embeddings_path)
all_embeddings = np.concatenate((train_embeddings, valid_embeddings))
norms = np.linalg.norm(all_embeddings, axis=1)
all_embeddings /= norms[:, np.newaxis]
print(all_embeddings.shape)
data_path = os.path.join(home, 'smart-news-query-embeddings/experiments', exp_name, 'data')
train_sentences_path = os.path.join(data_path, 'train_sentences.pkl')
valid_sentences_path = os.path.join(data_path, 'valid_sentences.pkl')
train_sentences = pd.read_pickle(train_sentences_path)
valid_sentences = pd.read_pickle(valid_sentences_path)
sentences = pd.concat((train_sentences, valid_sentences))
print(sentences.shape)

class PredictHandler(RequestHandler):    

    def get(self):
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
