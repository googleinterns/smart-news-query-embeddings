import pickle
import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import v_measure_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', '-e', type=str, required=True)
    parser.add_argument('--two-tower', '-t', action='store_true', default=False)
    args = parser.parse_args()
    home = os.path.expanduser('~')
    experiments_dir = os.path.join(home, 'smart-news-query-embeddings', 'experiments')
    exp_dir = os.path.join(experiments_dir, args.exp_name)
    embeddings_path = os.path.join(exp_dir, 'embeddings', 'valid_embeddings.npy')
    embeddings = np.load(embeddings_path)
    valid_data_path = os.path.join(exp_dir, 'data', 'valid_data.pkl')
    valid_labels_path = os.path.join(exp_dir, 'data', 'valid_labels.pkl')
    with open(valid_data_path, 'rb') as f:
        valid_data = pickle.load(f)
    with open(valid_labels_path, 'rb') as f:
        valid_labels = pickle.load(f)
    # model_path = os.path.join(exp_dir, 'model')
    # model = tf.keras.models.load_model(model_path)
    # pred_labels = model.predict(valid_data).argmax(axis=1)
    valid_labels = valid_labels.argmax(axis=1)
    # print('Validation Accuracy:', np.mean(pred_labels == valid_labels))
    N = valid_labels.shape[0]
    indices = np.random.choice(N, size=10000, replace=False)
    embeddings = embeddings[indices]
    valid_labels = valid_labels[indices]
    agg = AgglomerativeClustering(n_clusters=40).fit(embeddings)
    print('V-measure score (agglomerative clustering):', v_measure_score(valid_labels, agg.labels_))
    valid_df_path = os.path.join(exp_dir, 'data', 'valid_sentences.pkl')
    valid_sentences = pd.read_pickle(valid_df_path).iloc[indices]
    valid_df = pd.DataFrame({
        'abstract': valid_sentences,
        'label': agg.labels_
    })
    kmeans = KMeans(n_clusters=40).fit(embeddings)
    print('V-measure score (K-means clustering):', v_measure_score(valid_labels, kmeans.labels_))
    print(valid_df.shape)
    labels_out_path = os.path.join(exp_dir, 'data', 'embedding_labels.tsv')
    embeddings_out_path = os.path.join(exp_dir, 'data', 'embeddings.tsv')
    np.savetxt(embeddings_out_path, embeddings, delimiter='\t')
    valid_df.to_csv(labels_out_path, sep='\t')
