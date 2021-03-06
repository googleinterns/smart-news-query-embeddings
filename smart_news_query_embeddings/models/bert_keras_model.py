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

import tensorflow as tf
import bert
import os

class BertKerasModel(tf.keras.models.Model):

    def __init__(self, num_classes=2, bert_dir='uncased_L-12_H-768_A-12',
        max_seq_length=128, dense_size=256, dropout_rate=0.5, num_dense_layers=2,
        use_batch_norm=True):

        super(BertKerasModel, self).__init__()

        self.num_classes = num_classes
        self.bert_dir = bert_dir
        self.max_seq_length = max_seq_length
        self.dense_size = dense_size
        self.dropout_rate = dropout_rate
        self.num_dense_layers = num_dense_layers
        self.use_batch_norm = use_batch_norm # only used in the two-tower model, kept here for consistency.
        # ensures that self.get_embedding is serialized by model.save()
        self.get_embedding = tf.function(
            input_signature=[
                tf.TensorSpec(shape=(None, max_seq_length), dtype=tf.float32)
            ]
        )(self.get_embedding) # explicit form of tf.function decorator so we can use max_seq_length

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

    def get_embedding(self, x):
        out = x
        for layer in self.embedding_layers:
            out = layer(out)
        return out
