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

# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
import utils
import model_utils

"""
Wrapper class to set up BERT model. Takes in Pandas DataFrame
and is responsible for training, evaluating, and predicting.

Much of this code is adapted from the following notebook:

https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
"""
class BERTTrainer():

    # These hyperparameters are copied from this colab notebook
    # (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    def __init__(self, batch_size=32, max_seq_length=128,
            learning_rate=2e-5, num_train_epochs=3,
            warmup_proportion=0.1, save_checkpoints_every=500,
            save_summary_every=100, dropout_rate=0.5,
            is_training=True,
            bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            output_dir="bert_output"):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.save_checkpoints_every = save_checkpoints_every
        self.save_summary_every = save_summary_every
        self.dropout_rate = dropout_rate
        self.bert_model_hub = bert_model_hub
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir if os.path.exists(output_dir) else None
        self.is_training = is_training
        self.tokenizer = utils.create_tokenizer_from_hub_module(self.bert_model_hub)
        print('Saving models to {}'.format(output_dir))
        # self._train_and_test_features_from_df()

    def _create_estimator(self):
        # Compute train and warmup steps from batch size
        self.num_train_steps = int(
            len(self.train_features) / self.batch_size * self.num_train_epochs)
        self.num_warmup_steps = int(
            self.num_train_steps * self.warmup_proportion)

        # Specify output directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(
                model_dir=self.output_dir,
                save_summary_steps=self.save_summary_every,
                save_checkpoints_steps=self.save_checkpoints_every)

        model_fn = model_utils.model_fn_builder(
                bert_model_hub=self.bert_model_hub,
                num_labels=len(self.label_list),
                learning_rate=self.learning_rate,
                num_train_steps=self.num_train_steps,
                num_warmup_steps=self.num_warmup_steps,
                dropout_rate=self.dropout_rate)

        self.estimator = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config,
                warm_start_from=self.checkpoint_dir,
                params={"batch_size": self.batch_size})

    # Create an input function for training. drop_remainder = True for using TPUs.
    def train(self, train_inputs, train_labels):
        """ Train the BERT model.
        
        Arguments:
            train_inputs: Iterable of strings to train on.
            train_labels: Iterable of labels corresponding to each input.
        """

        self.train_features, self.label_list = utils.featurize_labeled_sentences(
            train_inputs, train_labels, self.tokenizer, self.max_seq_length)
        self._create_estimator()
        train_input_fn = bert.run_classifier.input_fn_builder(
                features=self.train_features,
                seq_length=self.max_seq_length,
                is_training=True,
                drop_remainder=False)

        print(f'Beginning Training!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)
        print("Training took time ", datetime.now() - current_time)

    def predict(self, inputs):
        """Predict classes for new inputs.
        Arguments:
            inputs: An iterable of string inputs to classify.
        Returns:
            A list of class labels corresponding to the input at
            each index.
        """
        input_features = utils.featurize_unlabeled_sentences(
            inputs, self.label_list, self.tokenizer, self.max_seq_length)
        predict_input_fn = run_classifier.input_fn_builder(
            features=input_features, seq_length=self.max_seq_length, is_training=False,
            drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)
        return [self.label_list[prediction['labels']] for prediction in predictions]
