import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from smart_news_query_embeddings.preprocessing.specificity_scores import \
get_spacy_responses

def generate_negatives_from_spacy_responses(token_lists, sentences, labels, ratio=1):

    token_lists = pd.Series(token_lists)

    # Filter out sentences with no named entities outputted by the NER.
    final_token_lists = token_lists[~token_lists.apply(lambda l: len(l) == 0)]

    # Group sentences by the tokens that appear in the sentence.
    # Do this manually, since it is possible for sentences to appear
    # in more than one group.
    type_indices = dict()

    for i in tqdm(final_token_lists.index):
        sentence = final_token_lists.loc[i]
        for token, token_type in sentence:
            if token_type not in type_indices:
                type_indices[token_type] = set()
            type_indices[token_type].add(i)

    # Calculate the relative frequencies of each entity type, and pass these to the function
    # that randomly samples entity types.
    keys = list(type_indices.keys())
    probs = np.array([len(type_indices[k]) for k in keys])
    prob_sum = probs.sum()
    if prob_sum == 0:
        return []
    probs /= prob_sum

    # Generate the negatives by taking pairs of sentences with at least
    # one token type in common occurring in both,
    # and having different labels, and swap their labels.
    negatives = []
    num_negs = int(ratio * len(sentences))
    pairs = set()
    print('Generating {} negatives'.format(num_negs))
    with tqdm(total=num_negs) as pbar:
        while len(negatives) < num_negs:
            key = np.random.choice(keys, p=probs)
            if len(type_indices[key]) < 2:
                continue
            i, j = random.sample(type_indices[key], 2)
            i, j = sorted([i, j])
            if (i, j) in pairs:
                continue
            sec1, sec2 = labels.iloc[i], labels.iloc[j]
            sent1, sent2 = sentences.iloc[i], sentences.iloc[j]
            pairs.add((i, j))
            if sec1 != sec2:
                negatives.extend([(sent1, sec2), (sent2, sec1)])
                pbar.update(2)
    return pd.DataFrame(negatives[:num_negs], columns=['sentence', 'label'])

def generate_negatives(sentences, labels, ratio=1):
    _, token_lists = get_spacy_responses(sentences)
    return generate_negatives_from_spacy_responses(token_lists, sentences, labels, ratio=ratio)