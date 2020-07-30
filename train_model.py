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
from bert_model_trainer import BertModelTrainer
from two_tower_model_trainer import TwoTowerModelTrainer
from bert_model_specificity_score_trainer import BertModelSpecificityScoreTrainer

if __name__ == '__main__':

    output_dir = 'bert_keras_output_{}'.format(int(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', default=32, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-5, type=float)
    parser.add_argument('--max-seq-length', default=128, type=int)
    parser.add_argument('--dropout-rate', default=0.5, type=float)
    parser.add_argument('--num-train-epochs', '-n', default=3, type=int)
    parser.add_argument('--dense-size', '-d', default=256, type=int)
    parser.add_argument('--exp-name', '-e', default=output_dir, type=str)
    parser.add_argument('--bert-dir', default='uncased_L-12_H-768_A-12', type=str)
    parser.add_argument('--two-tower', '-t', action='store_true', default=False)
    parser.add_argument('--specificity-scores', '-s', action='store_true', default=False)
    parser.add_argument('--cutoff', '-c', default=0.5, type=float)
    parser.add_argument('--train-tail', action='store_true', default=False)
    args = parser.parse_args()
    model_class = BertModelTrainer
    if args.two_tower:
        trainer = TwoTowerModelTrainer(args.exp_name, batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length, dropout_rate=args.dropout_rate, epochs=args.num_train_epochs,
        dense_size=args.dense_size, bert_dir=args.bert_dir)
    elif args.specificity_scores:
        trainer = BertModelSpecificityScoreTrainer(args.exp_name, batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length, dropout_rate=args.dropout_rate, epochs=args.num_train_epochs,
        dense_size=args.dense_size, bert_dir=args.bert_dir, tail_cutoff=args.cutoff, train_tail=args.train_tail)
    else:
        trainer = BertModelTrainer(args.exp_name, batch_size=args.batch_size, learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length, dropout_rate=args.dropout_rate, epochs=args.num_train_epochs,
        dense_size=args.dense_size, bert_dir=args.bert_dir)
    print('Trainer class is: {}'.format(type(trainer)))
    trainer.train()
