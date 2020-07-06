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

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import bert
import bert.run_classifier

def featurize_labeled_sentences(input_sentences, input_labels, tokenizer, max_seq_length):
    """Run BERT preprocessor on labeled sentences for training.

    Arguments:
        input_sentences: Iterable of strings to classify.
        input_labels: Iterable of labels associated with each string.
        tokenizer: The BERT preprocessor to use for tokenization.
        max_seq_length: Maximum number of tokens in input sequences.
    Returns:
        A list of InputFeature instances that can be consumed by the BERT model.
    """

    label_list = list(input_labels.unique())

    input_examples = [bert.run_classifier.InputExample(
        guid=None, # Globally unique ID for bookkeeping, unused in this example
        text_a=sentence,
        text_b=None,
        label=label
    ) for sentence, label in zip(input_sentences, input_labels)]

    # Convert our input examples to InputFeatures that BERT understands.
    input_features = bert.run_classifier.convert_examples_to_features(
        input_examples, label_list, max_seq_length, tokenizer)
    return input_features, label_list

def featurize_unlabeled_sentences(input_sentences, label_list, tokenizer, max_seq_length):
    """Run BERT preprocessor on input sentences with no labels, for prediction.

    Arguments:
        input_sentences: Iterable of strings to classify.
        label_list: List of possible labels, e.g. 0, 1, "dog", "cat", etc.
        tokenizer: The BERT preprocessor to use for tokenization.
        max_seq_length: Maximum number of tokens in input sequences.
    Returns:
        A list of InputFeature instances that can be consumed by the BERT model.
    """

    input_examples = [bert.run_classifier.InputExample(
        guid=None, # Globally unique ID for bookkeeping, unused in this example
        text_a=sentence,
        text_b=None,
        label=label_list[0]
    ) for sentence in input_sentences]

    # Convert our input examples to InputFeatures that BERT understands.
    input_features = bert.run_classifier.convert_examples_to_features(
        input_examples, label_list, max_seq_length, tokenizer)
    return input_features


def create_tokenizer_from_hub_module(bert_model_hub):
    """Get the vocab file and casing info from the Hub module.
    Arguments:
        bert_model_hub: The URL of the TF Hub BERT module to use.
    Returns:
        A tokenizer that can be used for preprocessing.
    """
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"]])

            return bert.tokenization.FullTokenizer(
                    vocab_file=vocab_file, do_lower_case=do_lower_case)
