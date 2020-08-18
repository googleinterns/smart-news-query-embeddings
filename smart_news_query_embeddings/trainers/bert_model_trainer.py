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

import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from smart_news_query_embeddings.models.bert_keras_model import BertKerasModel
from smart_news_query_embeddings.preprocessing.bert_tokenizer import \
create_tokenizer, get_filtered_nyt_data_with_scores, tokenize_data

"""
This class defines the trainer for the baseline model. It also provides an API for any
other models to be trained. The methods that would need to be overridden in a subclass
are the following:

get_data(self)
get_model(self)
get_train_and_valid_split(self, df)

It may additionally need to override the following properties:

train_x(self)
train_y(self)
valid_x(self)
valid_y(self)

Those are the methods and attributes that differ between
the models that we have implemented, and are appropriately subclassed
in bert_model_specificity_score_trainer.py and two_tower_model_trainer.py.
"""

class BertModelTrainer():

    DATA_PATH = 'data/nyt_articles_with_normalized_scores.pkl'
    RANDOM_SEED = 42
    INPUT_COLUMN = 'fixed_abstract'
    OUTPUT_COLUMN = 'section'
    BATCH_SIZE = 128

    def __init__(self, exp_name, max_seq_length=128, bert_dir='uncased_L-12_H-768_A-12',
        dense_size=256, learning_rate=1e-5, dropout_rate=0.5, epochs=3, batch_size=32,
        dry_run=False):
        self.exp_dir = os.path.join('experiments', exp_name)
        self.data_dir = os.path.join(self.exp_dir, 'data')
        self.out_dir = os.path.join(self.exp_dir, 'model')
        self.embeddings_dir = os.path.join(self.exp_dir, 'embeddings')
        self.max_seq_length = max_seq_length
        self.bert_dir = bert_dir
        self.dense_size = dense_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dry_run = dry_run
        if not os.path.exists('experiments'):
            os.mkdir('experiments')
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        self.train_data_path = os.path.join(self.data_dir, 'train_data.pkl')
        self.train_labels_path = os.path.join(self.data_dir, 'train_labels.pkl')
        self.valid_data_path = os.path.join(self.data_dir, 'valid_data.pkl')
        self.valid_labels_path = os.path.join(self.data_dir, 'valid_labels.pkl')
        self.train_sentences_path = os.path.join(self.data_dir, 'train_sentences.pkl')
        self.train_categories_path = os.path.join(self.data_dir, 'train_categories.pkl')
        self.test_sentences_path = os.path.join(self.data_dir, 'valid_sentences.pkl')
        self.test_categories_path = os.path.join(self.data_dir, 'valid_categories.pkl')
        self._load_data()
        self._load_model()

    def _load_data(self):
        if os.path.exists(self.train_data_path):
            self._get_data_from_cache()
        else:
            self.get_data()

    @property
    def train_x(self):
        return self.train_ids

    @property
    def train_y(self):
        return self.train_labels

    @property
    def valid_x(self):
        return self.test_ids

    @property
    def valid_y(self):
        return self.test_labels
    
    # Checks cache for pre-existing model, or instantiates a new one.
    def _load_model(self):
        if os.path.exists(self.out_dir):
            self.model = tf.keras.models.load_model(self.out_dir)
        else:
            self.get_model()

    # Instantiates a new model with all the hyperparameters provided
    # to this class.
    def get_model(self):
        self.model = BertKerasModel(self.num_classes, bert_dir=self.bert_dir,
            max_seq_length=self.max_seq_length, dense_size=self.dense_size,
            dropout_rate=self.dropout_rate)
        self.model.build(input_shape=(None, self.max_seq_length))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=self.learning_rate),
            metrics=['accuracy'])

    # Reads cached data. To avoid re-tokenization, which takes a lot of time.
    def _get_data_from_cache(self):
        with open(self.train_data_path, 'rb') as f:
            self.train_ids = pickle.load(f)
        with open(self.train_labels_path, 'rb') as f:
            self.train_labels = pickle.load(f)
        with open(self.valid_data_path, 'rb') as f:
            self.test_ids = pickle.load(f)
        with open(self.valid_labels_path, 'rb') as f:
            self.test_labels = pickle.load(f)
        self.num_classes = self.train_labels.shape[1]

    """
    How to split the data into training and validation sets.
    This varies between experiments so it may be necessary to
    override this functionality in subclasses.
    """
    def get_train_and_valid_split(self, df):
        return train_test_split(df, random_state=self.RANDOM_SEED)

    """
    Function to retrieve training and validation data, as well as parse it.
    This function is responsible for setting the following attributes:

    self.num_classes, self.train_ids, self.train_labels,
    self.valid_ids, self.valid_labels

    Any subclass must also define those 5 attributes in get_data if overridden.
    """
    def get_data(self):
        self.tokenizer = create_tokenizer(self.bert_dir)

        df = get_filtered_nyt_data_with_scores(self.DATA_PATH)
        if self.dry_run:
            df = df.sample(20)
        df['category_labels'] = df[self.OUTPUT_COLUMN].astype('category').cat.codes
        self.num_classes = df['category_labels'].max() + 1
        train_df, test_df = self.get_train_and_valid_split(df)
        train_sentences, train_categories = train_df[self.INPUT_COLUMN], train_df[self.OUTPUT_COLUMN]
        test_sentences, test_categories = test_df[self.INPUT_COLUMN], test_df[self.OUTPUT_COLUMN]
        train_sentences.to_pickle(self.train_sentences_path)
        test_sentences.to_pickle(self.test_sentences_path)
        train_categories.to_pickle(self.train_categories_path)
        test_categories.to_pickle(self.test_categories_path)
        self.train_ids, self.train_labels = tokenize_data(train_sentences,
            train_df['category_labels'], self.tokenizer, self.max_seq_length, self.num_classes)
        self.test_ids, self.test_labels = tokenize_data(test_sentences,
            test_df['category_labels'], self.tokenizer, self.max_seq_length, self.num_classes)
        with open(self.train_data_path, 'wb') as f:
            pickle.dump(self.train_ids, f)
        with open(self.train_labels_path, 'wb') as f:
            pickle.dump(self.train_labels, f)
        with open(self.valid_data_path, 'wb') as f:
            pickle.dump(self.test_ids, f)
        with open(self.valid_labels_path, 'wb') as f:
            pickle.dump(self.test_labels, f)

    def train(self):
        self.run_training()
        self.save_embeddings()

    def run_training(self):
        history = self.model.fit(x=self.train_x, y=self.train_y,
            validation_data=(self.valid_x, self.valid_y),
            epochs=self.epochs, batch_size=self.batch_size)
        history_path = os.path.join(self.data_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)

        self.model.save(self.out_dir)

    """
    Generate embeddings from the trained model. First generates embeddings for
    training data, then for validation data. Saves them within the experiment
    directory for later use.
    """
    def save_embeddings(self):
        if not os.path.exists(self.embeddings_dir):
            os.mkdir(self.embeddings_dir)
        print('Generating embeddings for training data...')
        train_embeddings = self.get_embeddings(self.train_ids)
        train_embeddings_path = os.path.join(self.embeddings_dir, 'train_embeddings.npy')
        np.save(train_embeddings_path, train_embeddings)
        print('Generating embeddings for validation data...')
        test_embeddings = self.get_embeddings(self.test_ids)
        test_embeddings_path = os.path.join(self.embeddings_dir, 'valid_embeddings.npy')
        np.save(test_embeddings_path, test_embeddings)

    """
    Computes embeddings using the trained model. Batch size of 128 is enough to ensure
    no out-of-memory issues on GCP. For ~60k examples, this takes about 2-3 minutes to run.
    """
    def get_embeddings(self, data):
        N = data.shape[0]
        embeddings = np.zeros((N, self.dense_size))
        for i in tqdm(range(0, N, self.BATCH_SIZE)): # tqdm shows a progress bar while this runs
            x = data[i:i + self.BATCH_SIZE]
            embeddings[i:i + self.BATCH_SIZE] = self.model.get_embedding(x).numpy()
        return embeddings
