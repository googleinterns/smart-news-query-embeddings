import os
import time
import argparse
import pickle
import numpy as np
import tensorflow as tf
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
    parser.add_argument('--exp-name', '-e', default=output_dir, type=str)
    parser.add_argument('--bert-dir', default='uncased_L-12_H-768_A-12', type=str)
    args = parser.parse_args()
    if not os.path.exists('output_data'):
        os.mkdir('output_data')
    exp_data_path = os.path.join('output_data', args.exp_name)
    if not os.path.exists(exp_data_path):
        os.mkdir(exp_data_path)
    train_data_path = os.path.join(exp_data_path, 'train_data.npy')
    train_labels_path = os.path.join(exp_data_path, 'train_labels.npy')
    valid_data_path = os.path.join(exp_data_path, 'valid_data.npy')
    valid_labels_path = os.path.join(exp_data_path, 'valid_labels.npy')
    if not os.path.exists('output_models'):
        os.mkdir('output_models')
    out_dir = os.path.join('output_models', args.exp_name)

    if not os.path.exists(train_data_path):
        all_train_ids = np.load('data/all_train_ids.npy')
        all_train_labels = np.load('data/all_train_labels.npy')
        all_train_outputs = np.load('data/all_train_outputs.npy')
        all_valid_ids = np.load('data/all_test_ids.npy')
        all_valid_labels = np.load('data/all_test_labels.npy')
        all_valid_outputs = np.load('data/all_test_outputs.npy')
    else:
        with open(train_data_path, 'rb') as f:
            (all_train_ids, all_train_labels) = pickle.load(f)
        with open(valid_data_path, 'rb') as f:
            (all_valid_ids, all_valid_labels) = pickle.load(f)
        with open(train_labels_path, 'rb') as f:
            all_train_labels = pickle.load(f)
        with open(valid_labels_path, 'rb') as f:
            all_valid_labels = pickle.load(f)

    with open(train_data_path, 'wb') as f:
        pickle.dump((all_train_ids, all_train_labels), f)
    with open(valid_data_path, 'wb') as f:
        pickle.dump((all_valid_ids, all_valid_labels), f)
    with open(train_labels_path, 'wb') as f:
        pickle.dump(all_train_labels, f)
    with open(valid_labels_path, 'wb') as f:
        pickle.dump(all_valid_labels, f)
    # np.save(train_labels_path, all_train_labels)
    # np.save(train_outputs_path, all_train_outputs)
    # np.save(valid_ids_path, (all_valid_ids, all_valid_labels))
    # np.save(valid_labels_path, all_valid_labels)
    # np.save(valid_outputs_path, all_valid_outputs)

    num_classes = all_train_labels.shape[1]

    if not os.path.exists(out_dir):
        model = TwoTowerModel(num_classes, bert_dir=args.bert_dir,
            max_seq_length=args.max_seq_length, dense_size=args.dense_size, dropout_rate=args.dropout_rate)

        model.build(input_shape=[(None, 128), (None, 64)])
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.learning_rate), metrics=['accuracy'])
    else:
        model = tf.keras.models.load_model(out_dir)
    print(model.summary())

    model.fit(x=(all_train_ids[:80], all_train_labels[:80]), y=all_train_outputs[:80],
    	validation_data=((all_valid_ids[:20], all_valid_labels[:20]), all_valid_outputs[:20]),
    	epochs=args.num_train_epochs,
    	batch_size=args.batch_size)

    model.save(out_dir)
