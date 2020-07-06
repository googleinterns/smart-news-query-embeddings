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

def create_model(bert_model_hub, is_predicting, input_ids, input_mask,
    segment_ids, labels, num_labels, dropout_rate):
        """ Creates a classification model.

        Args:
            bert_model_hub: URL of the TF Hub BERT module to use.
            is_predicting: Boolean to toggle training or testing mode.
            (input_ids, input_mask, segment_ids, labels): Output of
                bert.run_classifier.convert_examples_to_features
            num_labels: Number of classes to classify.
            dropout_rate: Keep probability of the dropout layer in the model.
        Returns:
            If training, loss and prediction tensors, otherwise just prediction tensors.
        """

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

def model_fn_builder(bert_model_hub, num_labels, learning_rate, num_train_steps,
        num_warmup_steps, dropout_rate):
    """Returns `model_fn` closure for TPUEstimator.
    Arguments:
        bert_model_hub: The URL of the TF Hub BERT module to use.
        num_labels: The number of classes to output.
        learning_rate: Learning rate during training.
        num_training_steps: Total number of steps during training across all epochs.
        num_warmup_steps: Number of steps with very low learning rate before starting training.
        dropout_rate: Keep probability in the dropout layer of the model.
    """
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

            (loss, predicted_labels, log_probs) = create_model(
                    bert_model_hub, is_predicting, input_ids, input_mask, segment_ids, label_ids,
                    num_labels, dropout_rate)

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
            (predicted_labels, log_probs) = create_model(
                    bert_model_hub, is_predicting, input_ids, input_mask, segment_ids, label_ids,
                    num_labels, dropout_rate)

            predictions = {
                    'probabilities': log_probs,
                    'labels': predicted_labels
                    }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn
