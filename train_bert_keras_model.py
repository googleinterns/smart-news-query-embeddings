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
import time
import argparse
import tensorflow as tf
from smart_news_query_embeddings.models.bert_keras_model import BertKerasModel
from smart_news_query_embeddings.preprocessing.bert_tokenizer import *
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/nyt_articles_with_normalized_scores.pkl'

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
    parser.add_argument('--output-dir', '-o', default=output_dir, type=str)
    parser.add_argument('--training', '-t', default=True, type=bool)
    parser.add_argument('--bert-dir', default='uncased_L-12_H-768_A-12', type=str)
    parser.add_argument('--tail-cutoff', default=0.5, type=float)
    args = parser.parse_args()
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    out_dir = os.path.join('outputs', args.output_dir)

    tokenizer = create_tokenizer(args.bert_dir)

    df = get_filtered_nyt_data_with_scores(DATA_PATH)
    df['category_labels'] = df['section'].astype('category').cat.codes
    num_classes = df['category_labels'].max() + 1
    CUTOFF = int(df.shape[0] * args.tail_cutoff)
    train_df, test_df = df[-CUTOFF:], df[:CUTOFF]
    train_ids, train_labels = tokenize_data(train_df['abstract'], train_df['category_labels'], tokenizer, args.max_seq_length, num_classes)
    test_ids, test_labels = tokenize_data(test_df['abstract'], test_df['category_labels'], tokenizer, args.max_seq_length, num_classes)
    if not os.path.exists(out_dir):
        model = BertKerasModel(num_classes, bert_dir=args.bert_dir,
            max_seq_length=args.max_seq_length, dense_size=args.dense_size,
            dropout_rate=args.dropout_rate)
        model.build(input_shape=(None, args.max_seq_length))
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.load_model(out_dir)
    
    model.fit(train_ids, train_labels, validation_data=(test_ids, test_labels), epochs=args.num_train_epochs, batch_size=args.batch_size)

    model.save(out_dir)
