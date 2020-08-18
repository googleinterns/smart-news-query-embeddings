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

import numpy as np
from smart_news_query_embeddings.models.two_tower_model import TwoTowerModel
from tensorflow.keras.optimizers import Adam
from smart_news_query_embeddings.trainers.bert_model_trainer import BertModelTrainer

class TwoTowerModelTrainer(BertModelTrainer):

    def get_data(self):
        self.train_ids = np.load('data/all_train_ids.npy')
        self.train_labels = np.load('data/all_train_labels.npy')
        self.train_outputs = np.load('data/all_train_outputs.npy')
        self.test_ids = np.load('data/all_test_ids.npy')
        self.test_labels = np.load('data/all_test_labels.npy')
        self.test_outputs = np.load('data/all_test_outputs.npy')
        if self.dry_run:
            train_indices = np.random.choice(self.train_ids.shape[0], size=15, replace=False)
            test_indices = np.random.choice(self.test_ids.shape[0], size=5, replace=False)
            self.train_ids = self.train_ids[train_indices]
            self.train_labels = self.train_labels[train_indices]
            self.train_outputs = self.train_outputs[train_indices]
            self.test_ids = self.test_ids[test_indices]
            self.test_labels = self.test_labels[test_indices]
            self.test_outputs = self.test_outputs[test_indices]
        self.num_classes = self.train_labels.shape[1]

    @property
    def train_x(self):
        return (self.train_ids, self.train_labels)

    @property
    def train_y(self):
        return self.train_outputs

    @property
    def valid_x(self):
        return (self.test_ids, self.test_labels)

    @property
    def valid_y(self):
        return self.test_outputs

    def get_model(self):
        self.model = TwoTowerModel(self.num_classes, bert_dir=self.bert_dir,
            max_seq_length=self.max_seq_length, dense_size=self.dense_size, dropout_rate=self.dropout_rate)

        self.model.build(input_shape=[(None, self.max_seq_length), (None, self.num_classes)])
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=self.learning_rate), metrics=['accuracy'])
