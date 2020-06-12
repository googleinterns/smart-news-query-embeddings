import unittest
import time
import pandas as pd
from bert_trainer import BERTTrainer
from utils import *

class TestBERT(unittest.TestCase):

    def test_init(self):
        trainer = BERTTrainer()

    def test_train(self):
        output_dir = 'test_{}'.format(str(int(time.time())))
        trainer = BERTTrainer(output_dir=output_dir)
        print(trainer.bert_model_hub)
        data = pd.DataFrame({
            'abstract': ['test one', 'test two', 'test three'] * 5,
            'section': ['U.S.', 'Arts', 'U.S.'] * 5,
        })
        data_column = 'abstract'
        label_column = 'section'
        max_seq_length = 128
        bert_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

        train_features, _, _, label_list = train_and_test_features_from_df(data, data_column, label_column, bert_model_hub, max_seq_length)
        trainer.train(train_features, label_list)

if __name__ == '__main__':
    unittest.main()
