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

import unittest
import pandas as pd
from utils import *
import bert.tokenization

class UtilTester(unittest.TestCase):

    bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

    def test_train_and_test_features_from_df(self):

        data = pd.DataFrame({
            'abstract': ['test one', 'test two', 'test three'] * 5,
            'section': ['U.S.', 'Arts', 'U.S.'] * 5,
        })

        N = data.shape[0]

        data_column = 'abstract'
        label_column = 'section'
        max_seq_length = 128

        train, test, tokenizer, label_list = train_and_test_features_from_df(data, data_column, label_column, self.bert_model_hub, max_seq_length)

        self.assertIsInstance(train, list)
        self.assertIsInstance(test, list)
        self.assertIsInstance(tokenizer, bert.tokenization.FullTokenizer)
        self.assertIsInstance(label_list, list)
        self.assertEqual(len(train), 4 * N // 5)
        self.assertEqual(len(test), N // 5)

if __name__ == '__main__':

    unittest.main()
