import csv
import os
import random
import numpy as np
import pandas as pd
import bert

from sklearn.model_selection import train_test_split

def get_filtered_nyt_data(data_path):
    print('Reading data...')
    df = pd.read_pickle(data_path)
    sections = df[['section', 'desk']].drop_duplicates()
    category_counts = sections.groupby('section').count().sort_values('desk', ascending=False)
    big_category_df = category_counts[category_counts['desk'] >= 20]

    big_categories = list(big_category_df.index)

    filtered = df[df['section'].isin(big_categories)]
    return filtered

def get_filtered_nyt_data_with_scores(data_path):
    print('Reading data...')
    df = pd.read_pickle(data_path)
    filtered_df = df[~df['normalized_abstract_score'].isnull()].drop_duplicates().sort_values('normalized_abstract_score')
    return filtered_df

def create_tokenizer(model_dir):
    vocab_file = os.path.join(model_dir, "vocab.txt")

    tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
    return tokenizer

def tokenize_data(inputs, labels, tokenizer, max_seq_length, num_classes):

    train_labels = []
    for l in labels:
        one_hot = [0] * num_classes
        one_hot[l] = 1
        train_labels.append(one_hot)

    train_tokens = map(tokenizer.tokenize, inputs)
    train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
    train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

    train_token_ids = map(
        lambda tids: tids + [0] * (max_seq_length - len(tids)) if len(tids) <= 128 else tids[:128],
        train_token_ids)
    train_token_ids = list(train_token_ids)
    print(type(train_token_ids), type(train_token_ids[0]))
    train_token_ids = np.array(train_token_ids)

    train_labels_final = np.array(train_labels)

    return train_token_ids, train_labels_final

if __name__ == '__main__':
    tokenizer = create_tokenizer('uncased_L-12_H-768_A-12')

    df = get_filtered_nyt_data('nyt_data_from_2015.pkl')
    df['category_labels'] = df['section'].astype('category').cat.codes
    print(df.head())
    train_df, test_df = train_test_split(df, random_state=42)
    train_ids, train_labels = tokenize_data(train_df['abstract'], train_df['category_labels'], tokenizer)
    test_ids, test_labels = tokenize_data(df['abstract'], train_df['category_labels'], tokenizer)
    print(train_ids.shape, train_labels.shape, train_ids[0])