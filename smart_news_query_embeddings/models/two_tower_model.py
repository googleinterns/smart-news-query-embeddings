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
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class TwoTowerModel(BertKerasModel):

    def build_model(self):
        # define two sets of inputs
        input_ids = Input(shape=(self.max_seq_length,), dtype='int32', name="input_ids")
        input_labels = Input(shape=(self.num_classes,), dtype='int32', name="input_labels")
        # the first branch operates on the first input
        bert_output = Flatten(name="flatten")(self.bert_layer(input_ids))
        x = Dense(self.dense_size, activation="relu", name="dense1_1")(bert_output)
        x = Dropout(0.5)(x)
        x = Dense(self.dense_size, activation="relu", name="dense1_2")(x)
        x = Dropout(0.5)(x)
        x = Model(inputs=input_ids, outputs=x, name="sub_model1")
        # the second branch opreates on the second input
        y = Dense(self.dense_size, activation="relu", name="dense2_1")(input_labels)
        y = Dense(self.dense_size, activation="relu", name="dense2_2")(y)
        y = Model(inputs=input_labels, outputs=y, name="sub_model2")
        # combine the output of the two branches
        combined = concatenate([x.output, y.output], name="concantenate")
        # apply a FC layer and then a classification prediction on the
        # combined outputs
        z = Dense(128, activation="relu", name="final_dense")(combined)
        z = Dense(2, activation="sigmoid", name="output_dense")(z)
        # our model will accept the inputs of the two branches and
        # then output a single value
        self.output_layer = Model(inputs=[x.input, y.input], outputs=z, name="two_tower_model")

    def call(self, inputs):
        return self.output_layer(inputs)
