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

from smart_news_query_embeddings.models.bert_keras_model import BertKerasModel
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class TwoTowerModel(BertKerasModel):

    def build_model(self):
        self.flatten = Flatten(name="flatten")
        self.bn = BatchNormalization(name="bn")
        self.dense1_1 = Dense(self.dense_size, activation="relu", name="dense1_1")
        self.dropout1 = Dropout(self.dropout_rate)
        self.dense1_2 = Dense(self.dense_size, activation="relu", name="dense1_2")
        self.dropout2 = Dropout(self.dropout_rate)

        self.dense2_1 = Dense(self.dense_size, activation="relu", name="dense2_1")
        self.dense2_2 = Dense(self.dense_size, activation="relu", name="dense2_2")

        self.fc_layer = Dense(128, activation="relu", name="final_dense")
        self.output_layer = Dense(2, activation="sigmoid", name="output_dense")
        self.pre_embedding_layers = [self.bert_layer, self.bn, self.flatten, self.dense1_1, self.dropout1]
        # define two sets of inputs
        # input_ids = Input(shape=(self.max_seq_length,), dtype='int32', name="input_ids")
        # input_labels = Input(shape=(self.num_classes,), dtype='int32', name="input_labels")
        # # the first branch operates on the first input
        # bert_output = Flatten(name="flatten")(self.bert_layer(input_ids))
        # x = Dense(self.dense_size, activation="relu", name="dense1_1")(bert_output)
        # x = Dropout(self.dropout_rate)(x)
        # x = Dense(self.dense_size, activation="relu", name="dense1_2")(x)
        # x = Dropout(self.dropout_rate)(x)
        # x = Model(inputs=input_ids, outputs=x, name="sub_model1")
        # # the second branch opreates on the second input
        # y = Dense(self.dense_size, activation="relu", name="dense2_1")(input_labels)
        # y = Dense(self.dense_size, activation="relu", name="dense2_2")(y)
        # y = Model(inputs=input_labels, outputs=y, name="sub_model2")
        # # combine the output of the two branches
        # combined = concatenate([x.output, y.output], name="concantenate")
        # # apply a FC layer and then a classification prediction on the
        # # combined outputs
        # z = Dense(128, activation="relu", name="final_dense")(combined)
        # z = Dense(2, activation="sigmoid", name="output_dense")(z)
        # # our model will accept the inputs of the two branches and
        # # then output a single value
        # self.output_layer = Model(inputs=[x.input, y.input], outputs=z, name="two_tower_model")

    @property
    def dense_embedding_weights(self):
        W, b = self.dense2_2.weights
        return W.numpy(), b.numpy()

    def call(self, inputs):
        input_ids, input_labels = inputs
        out1 = self.bert_layer(input_ids)
        out1 = self.bn(out1)
        out1 = self.flatten(out1)
        out1 = self.dense1_1(out1)
        out1 = self.dropout1(out1)
        out1 = self.dense1_2(out1)
        out1 = self.dropout2(out1)
        out1 = self.dense2_2(out1)
        out2 = self.dense2_1(input_labels)
        out2 = self.dense2_2(out2)
        out = concatenate([out1, out2])
        out = self.fc_layer(out)
        out = self.output_layer(out)
        return out
