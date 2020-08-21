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

"""
Reports the validation accuracy and V-measure score for BERT models.
Adapted from Colab notebook.
"""

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

    # Load the data and model from the experiment directory
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
    model_path = os.path.join(exp_dir, 'model')
    model = tf.keras.models.load_model(model_path)

    # Get validation accuracy by running predict and comparing the values with ground truth.
    pred_labels = model.predict(valid_data).argmax(axis=1)
    valid_labels = valid_labels.argmax(axis=1)
    print('Validation Accuracy:', np.mean(pred_labels == valid_labels))

    # Run clustering on the embeddings. Sample 10000 of the embeddings randomly because otherwise
    # the clustering algorithm crashes with out of memory issues.
    N = valid_labels.shape[0]
    indices = np.random.choice(N, size=10000, replace=False)
    embeddings = embeddings[indices]
    valid_labels = valid_labels[indices]

    # first agglomerative clustering
    agg = AgglomerativeClustering(n_clusters=40).fit(embeddings)
    print('V-measure score (agglomerative clustering):', v_measure_score(valid_labels, agg.labels_))
    valid_df_path = os.path.join(exp_dir, 'data', 'valid_sentences.pkl')
    valid_sentences = pd.read_pickle(valid_df_path).iloc[indices]
    valid_df = pd.DataFrame({
        'abstract': valid_sentences,
        'label': agg.labels_
    })

    # then K-means clustering
    kmeans = KMeans(n_clusters=40).fit(embeddings)
    print('V-measure score (K-means clustering):', v_measure_score(valid_labels, kmeans.labels_))
    print(valid_df.shape)

    # we save the clusters to a TSV format, as required by the tensorflow projector. This makes it
    # easy to upload and create visualizations there.
    labels_out_path = os.path.join(exp_dir, 'data', 'embedding_labels.tsv')
    embeddings_out_path = os.path.join(exp_dir, 'data', 'embeddings.tsv')
    np.savetxt(embeddings_out_path, embeddings, delimiter='\t')
    valid_df.to_csv(labels_out_path, sep='\t')
