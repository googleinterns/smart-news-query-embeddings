import tornado
import os
import json
from tornado.web import RequestHandler, Application
import tensorflow as tf
from smart_news_query_embeddings.preprocessing.bert_tokenizer import \
create_tokenizer, tokenize_data

home = os.path.expanduser('~')
bert_dir = os.path.join(home, 'smart-news-query-embeddings/uncased_L-12_H-768_A-12/')
tokenizer = create_tokenizer(bert_dir)
exp_dir = os.path.join(home, 'smart-news-query-embeddings/experiments', 'test_setup', 'model')
keras_model = tf.keras.models.load_model(exp_dir)

class PredictHandler(RequestHandler):    

    def get(self):
        article = self.get_argument('article', None, True)
        input_seq, _ = tokenize_data([article], [0], tokenizer, 128, 64)
        embedding = keras_model.get_embedding(input_seq).numpy()[0].astype(float)
        self.write(json.dumps(list(embedding)))

    def check_origin(self, origin):
        return True

if __name__ == '__main__':
    application = Application([
        (r"/predict", PredictHandler),
    ])
    application.listen(8888)
    print('STARTING APPLICATION ON LOCALHOST:8888')
    tornado.ioloop.IOLoop.current().start()