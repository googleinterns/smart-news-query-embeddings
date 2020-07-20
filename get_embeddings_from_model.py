import os
import time
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
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
    out_dir = os.path.join('outputs', args.output_dir)

    model = tf.keras.models.load_model(out_dir)
    W, b = model.dense.weights
    W, b = W.numpy(), b.numpy()
    print(W.shape, b.shape)
    tokenizer = create_tokenizer(args.bert_dir)

    df = get_filtered_nyt_data_with_scores(DATA_PATH)
    df['category_labels'] = df['section'].astype('category').cat.codes
    num_classes = df['category_labels'].max() + 1
    CUTOFF = int(df.shape[0] * args.tail_cutoff)
    train_df, test_df = df[-CUTOFF:], df[:CUTOFF]
    test_ids, test_labels = tokenize_data(test_df['abstract'], test_df['category_labels'], tokenizer, args.max_seq_length, num_classes)
    all_embeddings = np.zeros((0, 256))
    for i in tqdm(range(0, len(test_ids), 32)):
        embeddings = model.bert_layer(test_ids[i:i + 32])
        embeddings = model.flatten(embeddings)
        embeddings = tf.add(tf.matmul(embeddings, W), b)
        all_embeddings = np.concatenate((all_embeddings, embeddings), axis=0)
    print((all_embeddings < 0).any())
    np.save('data/all_embeddings.npy', all_embeddings)
    test_df.to_pickle('data/niche_test_df.pkl')
