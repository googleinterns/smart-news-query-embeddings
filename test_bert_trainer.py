import unittest
import time
import os
import pandas as pd
from bert_trainer import BERTTrainer
from utils import *

class TestBERT(unittest.TestCase):

    def test_init(self):
        self.output_dir = 'test_{}'.format(str(int(time.time())))
        self.trainer = BERTTrainer(output_dir=output_dir)

    def test_train(self):
        data = pd.DataFrame({
            'abstract': ['test one', 'test two', 'test three'] * 5,
            'section': ['U.S.', 'Arts', 'U.S.'] * 5,
        })
        data_column = 'abstract'
        label_column = 'section'

        train_features, test_features, _, label_list = train_and_test_features_from_df(data, data_column, label_column, trainer.bert_model_hub, trainer.max_seq_length)
        self.trainer.train(train_features, label_list)
        results = self.trainer.test(test_features)
        print('Evaluation results:', results)
        results2 = self.trainer.test(test_features)
        print('Evaluation results:', results2)
        eval_acc1, eval_acc2 = results['eval_accuracy'], results2['eval_accuracy']
        self.assertEqual(eval_acc1, eval_acc2)
        loss1, loss2 = results['loss'], results2['loss']
        self.assertEqual(eval_acc1, eval_acc2)
        os.rmdir(self.output_dir)

    def test_predict(self):
        input_sentences = [
            "test four",
            "test one",
        ] * 5
        preds = self.trainer.predict(input_sentences)

if __name__ == '__main__':
    unittest.main()
