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

        train_features, test_features, _, label_list = train_and_test_features_from_df(data, data_column, label_column, trainer.bert_model_hub, trainer.max_seq_length)
        trainer.train(train_features, label_list)
        results = trainer.test(test_features)
        print('Evaluation results:', results)

if __name__ == '__main__':
    unittest.main()
