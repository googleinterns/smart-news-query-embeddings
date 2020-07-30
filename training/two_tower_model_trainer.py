import numpy as np
from smart_news_query_embeddings.models.two_tower_model import TwoTowerModel
from tensorflow.keras.optimizers import Adam
from bert_model_trainer import BertModelTrainer

class TwoTowerModelTrainer(BertModelTrainer):

    def get_data(self):
        self.train_ids = np.load('data/all_train_ids.npy')
        self.train_labels = np.load('data/all_train_labels.npy')
        self.train_outputs = np.load('data/all_train_outputs.npy')
        self.test_ids = np.load('data/all_test_ids.npy')
        self.test_labels = np.load('data/all_test_labels.npy')
        self.test_outputs = np.load('data/all_test_outputs.npy')
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
