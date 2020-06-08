# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier

"""
Wrapper class to set up BERT model. Takes in Pandas DataFrame
and is responsible for training, evaluating, and predicting.

Much of this code is adapted from the following notebook:

https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
"""
class BERTTrainer():

    # These hyperparameters are copied from this colab notebook
    # (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    def __init__(self, data, batch_size=32, max_seq_length=128,
            learning_rate=2e-5, num_train_epochs=3, dense_size=256,
            warmup_proportion=0.1, save_checkpoints_every=500,
            save_summary_every=100,
            is_training=True,
            bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            output_dir="bert_output"):
        self.data = data
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.dense_size = dense_size
        self.warmup_proportion = warmup_proportion
        self.save_checkpoints_every = save_checkpoints_every
        self.save_summary_every = save_summary_every
        self.bert_model_hub = bert_model_hub
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir if os.path.exists(output_dir) else None
        self.is_training = is_training
        print('Saving models to {}'.format(output_dir))
        self._train_and_test_features_from_df()
        self._create_estimator()

    '''
    Compute 80-20 train-valid split from data, and tokenize/pad sequences
    into required BERT input form.
    '''

    def _train_and_test_features_from_df(self):
        train, test = train_test_split(self.data, test_size=0.2, random_state=42)
        print('Getting features for training and testing datasets')

        DATA_COLUMN = 'abstract'
        LABEL_COLUMN = 'section'
        self.label_list = list(self.data['section'].unique())

        train_input_examples = train.apply(lambda x: bert.run_classifier.InputExample(
            guid=None, # Globally unique ID for bookkeeping, unused in this example
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN]), axis=1)

        test_input_examples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
            text_a=x[DATA_COLUMN],
            text_b=None,
            label=x[LABEL_COLUMN]), axis=1)

        def _create_tokenizer_from_hub_module():
            """Get the vocab file and casing info from the Hub module."""
            with tf.Graph().as_default():
                bert_module = hub.Module(self.bert_model_hub)
                tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
                with tf.Session() as sess:
                    vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                        tokenization_info["do_lower_case"]])

                    return bert.tokenization.FullTokenizer(
                            vocab_file=vocab_file, do_lower_case=do_lower_case)

        self.tokenizer = _create_tokenizer_from_hub_module()

        # Convert our train and test features to InputFeatures that BERT understands.
        if self.is_training:
            self.train_features = bert.run_classifier.convert_examples_to_features(
                train_input_examples, self.label_list, self.max_seq_length, self.tokenizer)
            self.test_features = bert.run_classifier.convert_examples_to_features(
                test_input_examples, self.label_list, self.max_seq_length, self.tokenizer)

    def _create_model(self, is_predicting, input_ids, input_mask, segment_ids, labels,
            num_labels):
        """Creates a classification model."""

        bert_module = hub.Module(
                self.bert_model_hub,
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

        dense_weights = tf.get_variable(
                "dense_weights", [self.dense_size, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

        dense_bias = tf.get_variable(
                "dense_bias", [self.dense_size], initializer=tf.zeros_initializer())

        # Dense layer that outputs classes after BERT layer.
        output_weights = tf.get_variable(
                "output_weights", [num_labels, self.dense_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
                "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):

            # Dropout helps prevent overfitting
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            # Dense layer with ReLU activation after BERT
            dense_output = tf.matmul(output_layer, dense_weights, tranpose_b=True)
            dense_output = tf.add(dense_output, dense_bias)
            dense_output = tf.nn.relu(dense_output)

            logits = tf.matmul(dense_output, output_weights, transpose_b=True)
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

    # model_fn_builder actually creates our model function
    # using the passed parameters for num_labels, learning_rate, etc.
    def _model_fn_builder(self, num_labels, learning_rate, num_train_steps,
            num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator."""
        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:

                (loss, predicted_labels, log_probs) = self._create_model(
                        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                train_op = bert.optimization.create_optimizer(
                        loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

                # Calculate evaluation metrics.
                def metric_fn(label_ids, predicted_labels):
                    accuracy = tf.metrics.accuracy(label_ids, predicted_labels)
                    return {
                            "eval_accuracy": accuracy,
                            }

                eval_metrics = metric_fn(label_ids, predicted_labels)

                if mode == tf.estimator.ModeKeys.TRAIN:
                    return tf.estimator.EstimatorSpec(mode=mode,
                            loss=loss,
                            train_op=train_op)
                else:
                    return tf.estimator.EstimatorSpec(mode=mode,
                            loss=loss,
                            eval_metric_ops=eval_metrics)
            else:
                (predicted_labels, log_probs) = self._create_model(
                        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

                predictions = {
                        'probabilities': log_probs,
                        'labels': predicted_labels
                        }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Return the actual model function in the closure
        return model_fn


    def _create_estimator(self):
        # Compute train and warmup steps from batch size
        self.num_train_steps = int(
            len(self.train_features) / self.batch_size * self.num_train_epochs)
        self.num_warmup_steps = int(
            self.num_train_steps * self.warmup_proportion)

        # Specify output directory and number of checkpoint steps to save
        run_config = tf.estimator.RunConfig(
                model_dir=self.output_dir,
                save_summary_steps=self.save_summary_every,
                save_checkpoints_steps=self.save_checkpoints_every)

        model_fn = self._model_fn_builder(
                num_labels=len(self.label_list),
                learning_rate=self.learning_rate,
                num_train_steps=self.num_train_steps,
                num_warmup_steps=self.num_warmup_steps)

        self.estimator = tf.estimator.Estimator(
                model_fn=model_fn,
                config=run_config,
                warm_start_from=self.checkpoint_dir,
                params={"batch_size": self.batch_size})

    # Create an input function for training. drop_remainder = True for using TPUs.
    def train_model(self):
        train_input_fn = bert.run_classifier.input_fn_builder(
                features=self.train_features,
                seq_length=self.max_seq_length,
                is_training=True,
                drop_remainder=False)

        print(f'Beginning Training!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)
        print("Training took time ", datetime.now() - current_time)

    def test_model(self):
        test_input_fn = run_classifier.input_fn_builder(
                features=self.test_features,
                seq_length=self.max_seq_length,
                is_training=False,
                drop_remainder=False)

        return self.estimator.evaluate(input_fn=test_input_fn, steps=None)

    def predict(self, inputs):
        input_examples = [run_classifier.InputExample(
            guid="", text_a=x, text_b=None,
            label=self.label_list[0]) for x in inputs] # here, "" is just a dummy label
        input_features = run_classifier.convert_examples_to_features(
            input_examples, self.label_list, self.max_seq_length, self.tokenizer)
        predict_input_fn = run_classifier.input_fn_builder(
            features=input_features, seq_length=self.max_seq_length, is_training=False,
            drop_remainder=False)
        predictions = self.estimator.predict(predict_input_fn)
        return [self.label_list[prediction['labels']] for prediction in predictions]
