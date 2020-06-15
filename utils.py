from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import bert
import bert.run_classifier

def featurize_labeled_sentences(input_sentences, input_labels, tokenizer, max_seq_length):
    """Run BERT preprocessor and perform 80-20 train-test split on training data.

    Arguments:
        data: A Pandas DataFrame containing the training data.
        data_column: Name of the column containing the input sentences.
        label_column: Name of the column containing the class labels.
        bert_model_hub: URL of the BERT TF Hub module to use.
        max_seq_length: Maximum number of tokens to allow in the tokenized sequences.
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
    """Run BERT preprocessor and perform 80-20 train-test split on training data.

    Arguments:
        data: A Pandas DataFrame containing the training data.
        data_column: Name of the column containing the input sentences.
        label_column: Name of the column containing the class labels.
        bert_model_hub: URL of the BERT TF Hub module to use.
        max_seq_length: Maximum number of tokens to allow in the tokenized sequences.
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
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_model_hub)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                tokenization_info["do_lower_case"]])

            return bert.tokenization.FullTokenizer(
                    vocab_file=vocab_file, do_lower_case=do_lower_case)
