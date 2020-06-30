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

from bert_tokenizer import *
import os
import bert
import tensorflow as tf

class BertKerasModel():

    """Implementation of the BERT network using Keras.

    Arguments:

        num_classes: Required, number of classes to output probabilities for in the softmax layer.
        output_dir: Optional, name of directory to save trained model to.
        bert_dir: Path to downloaded BERT module.
        max_seq_length: Maximum number of tokens in input sequences.
        dense_size: Size of hidden dense layer between BERT output and softmax.
        learning_rate: Learning rate to use during training.
        epochs: Number of epochs to train for.
        batch_size: Batch size to use during training.
    """

    def __init__(self, num_classes, output_dir=None, bert_dir='uncased_L-12_H-768_A-12', max_seq_length=128,
        dense_size=256, learning_rate=1e-5, epochs=1, batch_size=32):

        self.num_classes = num_classes
        self.output_dir = output_dir
        self.bert_dir = bert_dir
        self.max_seq_length = max_seq_length
        self.dense_size = dense_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        if output_dir is not None and os.path.exists(output_dir):
            print('Loading saved model from {}'.format(os.path.abspath(output_dir)))
            self.model = tf.keras.models.load_model(output_dir)
        else:
            self.bert_layer = self._create_bert_layer(bert_dir)

            self.build_model()

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_seq_length,), dtype='int32', name='input_ids'),
            self.bert_layer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.dense_size, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax)
        ])

        self.model.build(input_shape=(None, self.max_seq_length))

        self.model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        print(self.model.summary())

    def _create_bert_layer(self, model_dir):
        bert_params = bert.params_from_pretrained_ckpt(model_dir)
        bert_layer = bert.BertModelLayer.from_params(bert_params, name="bert")
        bert_layer.apply_adapter_freeze()
        checkpoint_name = os.path.join(model_dir, "bert_model.ckpt.data-00000-of-00001")
        return bert_layer

    def fit(self, x, y, x_valid, y_valid):
        """Trains the model.

        Arguments:
            x: Training input sequences
            y: Training input labels
            x_valid: Validation input sequences
            y_valid: Validation input labels
        """

        self.model.fit(x, y, validation_data=(x_valid, y_valid), epochs=self.epochs, batch_size=self.batch_size)

        if self.output_dir is not None:
            self.model.save(self.output_dir)
