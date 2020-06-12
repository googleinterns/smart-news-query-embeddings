# -*- coding: utf-8 -*-

import os
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
import utils

"""
Wrapper class to set up BERT model. Takes in Pandas DataFrame
and is responsible for training, evaluating, and predicting.

Much of this code is adapted from the following notebook:

https://github.com/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
"""
class BERTTrainer():

    # These hyperparameters are copied from this colab notebook
    # (https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)
    def __init__(self, batch_size=32, max_seq_length=128,
            learning_rate=2e-5, num_train_epochs=3,
            warmup_proportion=0.1, save_checkpoints_every=500,
            save_summary_every=100, dropout_rate=0.5,
            is_training=True,
            bert_model_hub="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            output_dir="bert_output"):
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.save_checkpoints_every = save_checkpoints_every
        self.save_summary_every = save_summary_every
        self.dropout_rate = dropout_rate
        self.bert_model_hub = bert_model_hub
        self.output_dir = output_dir
        self.checkpoint_dir = output_dir if os.path.exists(output_dir) else None
        self.is_training = is_training
        print('Saving models to {}'.format(output_dir))
        # self._train_and_test_features_from_df()

    '''
    Compute 80-20 train-valid split from data, and tokenize/pad sequences
    into required BERT input form.
    '''

    """ model_fn_builder actually creates our model function using the passed parameters for num_labels, learning_rate, etc.

    Arguments:
        num_labels: The number of classes in our classification model.
        learning_rate: Float between 0 and 1 for learning rate.
        num_train_steps: An integer for the number of training steps to run.
        num_warmup_steps: An integer for the number of warmup steps to run before training.
    """
    def _model_fn_builder(self, num_labels, learning_rate, num_train_steps,
            num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator."""
        def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator.

            Arguments:
                features: list of features outputted by convert_examples_to_features
                labels: unused variable, required in model_fn signature.
                mode: are we training or testing?
                params: hyperparameters for the training process
            Returns:
                A tf.EstimatorSpec used internally by the estimator instance.
            """

            input_ids = features["input_ids"]
            input_mask = features["input_mask"]
            segment_ids = features["segment_ids"]
            label_ids = features["label_ids"]

            is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

            # TRAIN and EVAL
            if not is_predicting:

                print(self.bert_model_hub)
                (loss, predicted_labels, log_probs) = utils.create_model(
                        self.bert_model_hub, is_predicting, input_ids, input_mask, segment_ids, label_ids,
                        num_labels, self.dropout_rate)

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
                (predicted_labels, log_probs) = utils.create_model(
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
    def train(self, train_features, label_list):
        self.train_features = train_features
        self.label_list = label_list
        self._create_estimator()
        train_input_fn = bert.run_classifier.input_fn_builder(
                features=self.train_features,
                seq_length=self.max_seq_length,
                is_training=True,
                drop_remainder=False)

        print(f'Beginning Training!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)
        print("Training took time ", datetime.now() - current_time)

    def test(self, test_features):
        test_input_fn = run_classifier.input_fn_builder(
                features=test_features,
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
