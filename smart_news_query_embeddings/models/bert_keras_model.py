import tensorflow as tf
import bert
import os

class BertKerasModel(tf.keras.models.Model):

    def __init__(self, num_classes=2, bert_dir='uncased_L-12_H-768_A-12',
        max_seq_length=128, dense_size=256, dropout_rate=0.5, num_dense_layers=2):

        super(BertKerasModel, self).__init__()

        self.num_classes = num_classes
        self.bert_dir = bert_dir
        self.max_seq_length = max_seq_length
        self.dense_size = dense_size
        self.dropout_rate = dropout_rate
        self.num_dense_layers = num_dense_layers

        self.bert_layer = self._create_bert_layer()
        self.build_model()

    def build_model(self):
        self.flatten = tf.keras.layers.Flatten()
        self.dense_layers = []
        for _ in range(self.num_dense_layers):
            self.dense_layers.append(tf.keras.layers.Dense(self.dense_size))
            self.dense_layers.append(tf.keras.layers.LeakyReLU())
            self.dense_layers.append(tf.keras.layers.Dropout(self.dropout_rate))
        self.embedding_layers = [self.bert_layer, self.flatten] + self.dense_layers[:-2]
        self.output_layer = tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax)

    def _create_bert_layer(self):
        # Loads a BERT Keras layer from a downloaded pretrained module.
        bert_params = bert.params_from_pretrained_ckpt(self.bert_dir)
        bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
        bert_layer.apply_adapter_freeze()
        checkpoint_name = os.path.join(self.bert_dir, "bert_model.ckpt.data-00000-of-00001")
        return bert_layer

    def call(self, inputs):
        out = self.bert_layer(inputs)
        out = self.flatten(out)
        for layer in self.dense_layers:
            out = layer(out)
        out = self.output_layer(out)
        return out

    @tf.function
    def get_embedding(self, x):
        out = x
        for layer in self.embedding_layers:
            out = layer(out)
        return out
