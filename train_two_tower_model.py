import os
import time
import argparse
import numpy as np
from smart_news_query_embeddings.models.two_tower_model import TwoTowerModel
from tensorflow.keras.optimizers import Adam

if __name__ == '__main__':
    output_dir = 'bert_keras_output_{}'.format(int(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', default=32, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-5, type=float)
    parser.add_argument('--max-seq-length', default=128, type=int)
    parser.add_argument('--dropout-rate', default=0.5, type=float)
    parser.add_argument('--num-train-epochs', '-n', default=3, type=int)
    parser.add_argument('--dense-size', '-d', default=256, type=int)
    parser.add_argument('--output-dir', '-o', default=output_dir, type=str)
    parser.add_argument('--bert-dir', default='uncased_L-12_H-768_A-12', type=str)
    args = parser.parse_args()

    all_train_ids = np.load('data/all_train_ids.npy')
    all_train_labels = np.load('data/all_train_labels.npy')
    all_train_outputs = np.load('data/all_train_outputs.npy')
    all_test_ids = np.load('data/all_test_ids.npy')
    all_test_labels = np.load('data/all_test_labels.npy')
    all_test_outputs = np.load('data/all_test_outputs.npy')

    num_classes = all_train_labels.shape[1]

    model = TwoTowerModel(num_classes, bert_dir=args.bert_dir, max_seq_length=args.max_seq_length, dense_size=args.dense_size)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])

    model.fit(x=(all_train_ids, all_train_labels), y=all_train_outputs,
    	validation_data=((all_test_ids, all_test_labels), all_test_outputs),
    	epochs=args.num_train_epochs,
    	batch_size=args.batch_size)

    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    out_dir = os.path.join('outputs', args.output_dir)
    model.save(out_dir)
