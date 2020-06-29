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

import time
import argparse
from bert_tokenizer import *
from sklearn.model_selection import train_test_split
from bert_keras_layer import BertKerasModel
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class TwoTowerModel(BertKerasModel):

    def build_model(self):
        # define two sets of inputs
        input_ids = Input(shape=(self.max_seq_length,), dtype='int32', name="input_ids")
        input_labels = Input(shape=(self.num_classes,), dtype='int32', name="input_labels")
        # the first branch operates on the first input
        bert_output = Flatten(name="flatten")(self.bert_layer(input_ids))
        x = Dense(self.dense_size, activation="relu", name="dense1_1")(bert_output)
        x = Dense(self.dense_size, activation="relu", name="dense1_2")(x)
        x = Model(inputs=input_ids, outputs=x, name="sub_model1")
        # the second branch opreates on the second input
        y = Dense(self.dense_size, activation="relu", name="dense2_1")(input_labels)
        y = Dense(self.dense_size, activation="relu", name="dense2_2")(y)
        y = Model(inputs=input_labels, outputs=y, name="sub_model2")
        # combine the output of the two branches
        combined = concatenate([x.output, y.output], name="concantenate")
        # apply a FC layer and then a classification prediction on the
        # combined outputs
        z = Dense(128, activation="relu", name="final_dense")(combined)
        z = Dense(2, activation="sigmoid", name="output_dense")(z)
        # our model will accept the inputs of the two branches and
        # then output a single value
        self.model = Model(inputs=[x.input, y.input], outputs=z, name="two_tower_model")
        self.model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

        print(self.model.summary())

    def fit(self, train_inputs, train_labels, train_outputs, valid_inputs, valid_labels, valid_outputs):
        super().fit((train_inputs, train_labels), train_outputs, (valid_inputs, valid_labels), valid_outputs)

# def create_tokenizer(model_dir):

#     """Creates a BERT pre-processor from a module.

#     Arguments:
#         model_dir: Path to the downloaded BERT module.

#     Returns:
#         A BERT tokenizer that can be used to generate input sequences.
#     """

#     vocab_file = os.path.join(model_dir, "vocab.txt")

#     tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
#     return tokenizer

# tokenizer = create_tokenizer('uncased_L-12_H-768_A-12')

# def tokenize_data(inputs, labels, tokenizer, max_seq_length, num_classes):

#     """Creates input sequences and one-hot encoded labels from raw input.

#     Arguments:
#         inputs: List of input sentences.
#         labels: List of input labels as integers.
#         tokenizer: A BERT tokenizer as instantiated from create_tokenizer().
#         max_seq_length: Maximum number of tokens to include in the sequences.
#         num_classes: Total number of classes in the input labels.
#     """

#     train_labels = []
#     for l in labels:
#         one_hot = [0] * num_classes
#         one_hot[l] = 1
#         train_labels.append(one_hot)

#     train_tokens = map(tokenizer.tokenize, inputs)
#     train_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], train_tokens)
#     train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

#     train_token_ids = map(
#         lambda tids: tids + [0] * (max_seq_length - len(tids)) if len(tids) <= 128 else tids[:128],
#         train_token_ids)
#     train_token_ids = list(train_token_ids)
#     print(type(train_token_ids), type(train_token_ids[0]))
#     train_token_ids = np.array(train_token_ids)

#     train_labels_final = np.array(train_labels)

#     return train_token_ids, train_labels_final

# MAX_SEQ_LENGTH = 128
# NUM_CLASSES = 64

# def get_filtered_nyt_data_with_scores(data_path):

#     """Reads in the NYT article data containing specificity scores.

#     Arguments:
#         data_path: Path to the data pickle file.

#     Returns:
#         The filtered NYT data with only unique rows that have a specificity score included.
#     """

#     print('Reading data...')
#     df = pd.read_pickle(data_path)
#     filtered_df = df[~df['normalized_abstract_score'].isnull()].drop_duplicates().sort_values('normalized_abstract_score')
#     return filtered_df

if __name__ == '__main__':
    output_dir = 'bert_keras_output_{}'.format(int(time.time()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', '-b', default=32, type=int)
    parser.add_argument('--learning-rate', '-l', default=1e-5, type=float)
    parser.add_argument('--max-seq-length', default=128, type=int)
    parser.add_argument('--warmup-proportion', default=0.1, type=float)
    parser.add_argument('--dropout-rate', default=0.5, type=float)
    parser.add_argument('--num-train-epochs', '-n', default=3, type=int)
    parser.add_argument('--dense-size', '-d', default=256, type=int)
    parser.add_argument('--save-summary-every', default=100, type=int)
    parser.add_argument('--output-dir', '-o', default=output_dir, type=str)
    parser.add_argument('--training', '-t', default=True, type=bool)
    parser.add_argument('--bert-dir', default='uncased_L-12_H-768_A-12', type=str)
    args = parser.parse_args()

    all_train_ids = np.load('all_train_ids.npy')
    all_train_labels = np.load('all_train_labels.npy')
    all_train_outputs = np.load('all_train_outputs.npy')
    all_test_ids = np.load('all_test_ids.npy')
    all_test_labels = np.load('all_test_labels.npy')
    all_test_outputs = np.load('all_test_outputs.npy')

    num_classes = all_train_labels.shape[1]

    model = TwoTowerModel(num_classes, bert_dir=args.bert_dir, output_dir=args.output_dir, batch_size=args.batch_size,
        epochs=args.num_train_epochs, max_seq_length=args.max_seq_length, dense_size=args.dense_size,
        learning_rate=args.learning_rate)

    model.fit(all_train_ids, all_train_labels, all_train_outputs, all_test_ids, all_test_labels, all_test_outputs)
