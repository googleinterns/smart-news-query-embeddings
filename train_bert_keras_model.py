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
import sys
import time
import argparse
import pickle
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from smart_news_query_embeddings.models.bert_keras_model import BertKerasModel
from smart_news_query_embeddings.preprocessing.bert_tokenizer import *
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/nyt_articles_with_normalized_scores.pkl'
RANDOM_SEED = 42

def get_embeddings(model, data):
    batch_size = 128
    N = data.shape[0]
    embeddings = np.zeros((N, 256))
    for i in tqdm(range(0, N, batch_size)):
        x = data[i:i + batch_size]
        embeddings[i:i + batch_size] = model.get_embedding(x).numpy()
    return embeddings

if __name__ == '__main__':

    output_dir = 'bert_keras_output_{}'.format(int(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', default=32, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-5, type=float)
    parser.add_argument('--max-seq-length', default=128, type=int)
    parser.add_argument('--warmup-proportion', default=0.1, type=float)
    parser.add_argument('--dropout-rate', default=0.5, type=float)
    parser.add_argument('--num-train-epochs', '-n', default=3, type=int)
    parser.add_argument('--dense-size', '-d', default=256, type=int)
    parser.add_argument('--save-summary-every', default=100, type=int)
    parser.add_argument('--exp-name', '-e', default=output_dir, type=str)
    parser.add_argument('--training', '-t', default=True, type=bool)
    parser.add_argument('--bert-dir', default='uncased_L-12_H-768_A-12', type=str)
    parser.add_argument('--tail-cutoff', default=0.5, type=float)
    args = parser.parse_args()
    exp_dir = os.path.join('experiments', args.exp_name)
    data_dir = os.path.join(exp_dir, 'data')
    out_dir = os.path.join(exp_dir, 'model')
    if not os.path.exists('experiments'):
        os.mkdir('experiments')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    train_data_path = os.path.join(data_dir, 'train_data.pkl')
    train_labels_path = os.path.join(data_dir, 'train_labels.pkl')
    valid_data_path = os.path.join(data_dir, 'valid_data.pkl')
    valid_labels_path = os.path.join(data_dir, 'valid_labels.pkl')
    train_sentences_path = os.path.join(data_dir, 'train_sentences.pkl')
    train_categories_path = os.path.join(data_dir, 'train_categories.pkl')
    test_sentences_path = os.path.join(data_dir, 'valid_sentences.pkl')
    test_categories_path = os.path.join(data_dir, 'valid_categories.pkl')

    if not os.path.exists(train_data_path):
        tokenizer = create_tokenizer(args.bert_dir)

        df = get_filtered_nyt_data_with_scores(DATA_PATH).sample(100)
        df['category_labels'] = df['section'].astype('category').cat.codes
        num_classes = df['category_labels'].max() + 1
        CUTOFF = int(df.shape[0] * args.tail_cutoff)
        train_df, test_df = train_test_split(df, random_state=RANDOM_SEED)
        train_sentences, train_categories = train_df['fixed_headline'], train_df['section']
        test_sentences, test_categories = test_df['fixed_headline'], test_df['section']
        train_sentences.to_pickle(train_sentences_path)
        test_sentences.to_pickle(test_sentences_path)
        train_categories.to_pickle(train_categories_path)
        test_categories.to_pickle(test_categories_path)
        train_ids, train_labels = tokenize_data(train_sentences, train_df['category_labels'], tokenizer, args.max_seq_length, num_classes)
        test_ids, test_labels = tokenize_data(test_sentences, test_df['category_labels'], tokenizer, args.max_seq_length, num_classes)
        with open(train_data_path, 'wb') as f:
            pickle.dump(train_ids, f)
        with open(train_labels_path, 'wb') as f:
            pickle.dump(train_labels, f)
        with open(valid_data_path, 'wb') as f:
            pickle.dump(test_ids, f)
        with open(valid_labels_path, 'wb') as f:
            pickle.dump(test_labels, f)
    else:
        with open(train_data_path, 'rb') as f:
            train_ids = pickle.load(f)
        with open(train_labels_path, 'rb') as f:
            train_labels = pickle.load(f)
        with open(valid_data_path, 'rb') as f:
            test_ids = pickle.load(f)
        with open(valid_labels_path, 'rb') as f:
            test_labels = pickle.load(f)
        num_classes = train_labels.shape[1]
    if not os.path.exists(out_dir):
        model = BertKerasModel(num_classes, bert_dir=args.bert_dir,
            max_seq_length=args.max_seq_length, dense_size=args.dense_size,
            dropout_rate=args.dropout_rate)
        model.build(input_shape=(None, args.max_seq_length))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.load_model(out_dir)

    print(model.summary())
    print(train_ids.shape, train_labels.shape, test_ids.shape, test_labels.shape)
    
    history = model.fit(train_ids, train_labels, validation_data=(test_ids, test_labels), epochs=args.num_train_epochs, batch_size=args.batch_size)
    history_path = os.path.join(data_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)

    model.save(out_dir)
    embeddings_dir = os.path.join(exp_dir, 'embeddings')
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)
    print('Generating embeddings for training data...')
    train_embeddings = get_embeddings(model, train_ids)
    train_embeddings_path = os.path.join(embeddings_dir, 'train_embeddings.npy')
    np.save(train_embeddings_path, train_embeddings)
    print('Generating embeddings for validation data...')
    test_embeddings = get_embeddings(model, test_ids)
    test_embeddings_path = os.path.join(embeddings_dir, 'valid_embeddings.npy')
    np.save(test_embeddings_path, test_embeddings)
