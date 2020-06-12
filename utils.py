from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import bert
import bert.run_classifier

def train_and_test_features_from_df(data, data_column, label_column, bert_model_hub, max_seq_length):
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    label_list = list(data[label_column].unique())

    train_input_examples = train.apply(lambda x: bert.run_classifier.InputExample(
        guid=None, # Globally unique ID for bookkeeping, unused in this example
        text_a=x[data_column],
        text_b=None,
        label=x[label_column]), axis=1)

    test_input_examples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
        text_a=x[data_column],
        text_b=None,
        label=x[label_column]), axis=1)

    def _create_tokenizer_from_hub_module():
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():
            bert_module = hub.Module(bert_model_hub)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                    tokenization_info["do_lower_case"]])

                return bert.tokenization.FullTokenizer(
                        vocab_file=vocab_file, do_lower_case=do_lower_case)

    tokenizer = _create_tokenizer_from_hub_module()

    # Convert our train and test features to InputFeatures that BERT understands.
    train_features = bert.run_classifier.convert_examples_to_features(
        train_input_examples, label_list, max_seq_length, tokenizer)
    test_features = bert.run_classifier.convert_examples_to_features(
        test_input_examples, label_list, max_seq_length, tokenizer)
    return train_features, test_features, tokenizer, label_list

def create_model(bert_model_hub, is_predicting, input_ids, input_mask,
    segment_ids, labels, num_labels, dropout_rate):
        """ Creates a classification model.

        Args:
            is_predicting: Boolean to toggle training or testing mode.
            (input_ids, input_mask, segment_ids, labels): Output of
                bert.run_classifier.convert_examples_to_features
            num_labels: Number of classes to classify.
        Returns:
            If training, loss and prediction tensors, otherwise just prediction tensors.
        """

        print(bert_model_hub)
        bert_module = hub.Module(
                bert_model_hub,
                trainable=True)
        bert_inputs = dict(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids)
        bert_outputs = bert_module(
                inputs=bert_inputs,
                signature="tokens",
                as_dict=True)

        # Use "pooled_output" for classification tasks on an entire sentence.
        # Use "sequence_outputs" for token-level output.
        output_layer = bert_outputs["pooled_output"]

        hidden_size = output_layer.shape[-1].value

        # Dense layer that outputs classes after BERT layer.
        output_weights = tf.get_variable(
                "output_weights", [num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

        print(output_layer.shape, output_weights.shape)

        output_bias = tf.get_variable(
                "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):

            # Dropout helps prevent overfitting during training
            output_layer = tf.nn.dropout(output_layer, keep_prob=dropout_rate, seed=42)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.add(logits, output_bias)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # Convert labels into one-hot encoding
            one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

            predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
            # If we're predicting, we want predicted labels and the probabiltiies.
            if is_predicting:
                return (predicted_labels, log_probs)

            # If we're train/eval, compute loss between predicted and actual label
            per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
            loss = tf.reduce_mean(per_example_loss)
            return (loss, predicted_labels, log_probs)
