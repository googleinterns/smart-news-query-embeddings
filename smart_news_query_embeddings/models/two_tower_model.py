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
from smart_news_query_embeddings.models.bert_keras_model import BertKerasModel
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, concatenate, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class TwoTowerModel(BertKerasModel):

    def build_model(self):
        self.flatten = Flatten(name="flatten")
        self.dense1_1 = Dense(self.dense_size, name="dense1_1")
        self.bn1 = BatchNormalization(name="bn1")
        self.relu1_1 = LeakyReLU(name="relu1_1")
        self.dropout1 = Dropout(self.dropout_rate)
        self.dense1_2 = Dense(self.dense_size, name="dense1_2")
        self.bn2 = BatchNormalization(name="bn2")
        self.relu1_2 = LeakyReLU(name="relu1_2")
        self.dropout2 = Dropout(self.dropout_rate)

        self.dense2_1 = Dense(self.dense_size, name="dense2_1")
        self.relu2_1 = LeakyReLU(name="relu2_1")
        self.dense2_2 = Dense(self.dense_size, name="dense2_2")
        self.relu2_2 = LeakyReLU(name="relu2_2")

        self.final_dense = Dense(128, name="final_dense")
        self.final_relu = LeakyReLU(name="final_relu")
        self.output_layer = Dense(2, activation="sigmoid", name="output_dense")
        self.embedding_layers = [
            self.bert_layer,
            self.flatten,
            self.dense1_1,
            self.bn1,
            self.relu1_1,
            self.dropout1,
            self.dense1_2,
            self.bn2
        ]

    def call(self, inputs):
        input_ids, input_labels = inputs
        out1 = self.bert_layer(input_ids)
        out1 = self.flatten(out1)
        out1 = self.dense1_1(out1)
        out1 = self.bn1(out1)
        out1 = self.relu1_1(out1)
        out1 = self.dropout1(out1)
        out1 = self.dense1_2(out1)
        out1 = self.bn2(out1)
        out1 = self.relu1_2(out1)
        out1 = self.dropout2(out1)
        out2 = self.dense2_1(input_labels)
        out2 = self.relu2_1(out2)
        out2 = self.dense2_2(out2)
        out2 = self.relu2_2(out2)
        out = concatenate([out1, out2])
        out = self.final_dense(out)
        out = self.final_relu(out)
        out = self.output_layer(out)
        return out
