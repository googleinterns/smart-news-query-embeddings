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

import csv
import os
import random
import numpy as np
import pandas as pd
import bert
from tqdm import tqdm

from sklearn.model_selection import train_test_split

CATEGORY_THRESHOLD = 20
RANDOM_SEED = 42
CLS = '[CLS]'
SEP = '[SEP]'

def get_filtered_nyt_data(data_path):

    """Reads in the NYT article data.

    Arguments:
        data_path: Path to the data pickle file.

    Returns:
        The filtered NYT data with only major categories included.
    """

    print('Reading data...')
    df = pd.read_pickle(data_path)
    sections = df[['section', 'desk']].drop_duplicates() # desk represents a more specific subcategory of the articles
    category_counts = sections.groupby('section').count().sort_values('desk', ascending=False)
    big_category_df = category_counts[category_counts['desk'] >= CATEGORY_THRESHOLD]

    big_categories = list(big_category_df.index)

    filtered = df[df['section'].isin(big_categories)]
    return filtered

def get_filtered_nyt_data_with_scores(data_path):

    """Reads in the NYT article data containing specificity scores.

    Arguments:
        data_path: Path to the data pickle file.

    Returns:
        The filtered NYT data with only unique rows that have a specificity score included.
    """

    print('Reading data...')
    df = pd.read_pickle(data_path)
    filtered_df = df[~df['normalized_abstract_score'].isnull()].drop_duplicates().sort_values('normalized_abstract_score')
    return filtered_df

def create_tokenizer(model_dir):

    """Creates a BERT pre-processor from a module.

    Arguments:
        model_dir: Path to the downloaded BERT module.

    Returns:
        A BERT tokenizer that can be used to generate input sequences.
    """

    vocab_file = os.path.join(model_dir, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

def tokenize_data(inputs, labels, tokenizer, max_seq_length, num_classes):

    """Creates input sequences and one-hot encoded labels from raw input.

    Arguments:
        inputs: List of input sentences.
        labels: List of input labels as integers.
        tokenizer: A BERT tokenizer as instantiated from create_tokenizer().
        max_seq_length: Maximum number of tokens to include in the sequences.
        num_classes: Total number of classes in the input labels.
    """

    train_labels = []
    for l in labels:
        one_hot = [0] * num_classes
        one_hot[l] = 1
        train_labels.append(one_hot)

    print('Tokenizing {} inputs'.format(len(inputs)))
    train_token_ids = []
    for inp in tqdm(inputs):
        tokens = [CLS] + tokenizer.tokenize(inp) + [SEP]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        ids = ids + [0] * (max_seq_length - len(ids)) \
        if len(ids) <= max_seq_length else ids[:max_seq_length]
        train_token_ids.append(ids)
    train_token_ids = np.array(train_token_ids)

    train_labels_final = np.array(train_labels)

    return train_token_ids, train_labels_final

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', defualt='nyt_data_from_2015.pkl', type=str)

    args = parser.parse_args()

    tokenizer = create_tokenizer('uncased_L-12_H-768_A-12')

    # Read data in from pickled NYT articles and filter by articles with categories that
    # only have 20 or more subcategories. This narrows down the number of classes in the classification problem
    # which improves accuracy.
    df = get_filtered_nyt_data(args.data_path)
    df['category_labels'] = df['section'].astype('category').cat.codes
    print(df.head())
    train_df, test_df = train_test_split(df, random_state=RANDOM_SEED)
    # 'abstract' column has the abstract of an article (input sentence) and 'category_labels'
    # has a numeric value for the category ID of that article.
    train_ids, train_labels = tokenize_data(train_df['abstract'], train_df['category_labels'], tokenizer)
    test_ids, test_labels = tokenize_data(df['abstract'], train_df['category_labels'], tokenizer)
    np.save('data/filtered_train_ids.npy', train_ids)
    np.save('data/filtered_test_ids.npy', test_ids)
    np.save('data/filtered_train_labels.npy', train_labels)
    np.save('data/filtered_test_labels.npy', test_labels)
