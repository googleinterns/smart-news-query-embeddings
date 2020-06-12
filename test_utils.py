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
