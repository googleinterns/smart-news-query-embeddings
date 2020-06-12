import unittest
import time
import shutil
import pandas as pd
from bert_trainer import BERTTrainer
from utils import *

class TestBERT(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestBERT, self).__init__(*args, **kwargs)
        self.output_dir = 'test_{}'.format(str(int(time.time())))
        self.trainer = BERTTrainer(output_dir=self.output_dir)

    def train_model(self):
        data = pd.DataFrame({
            'abstract': ['test one', 'test two', 'test three'] * 5,
            'section': ['U.S.', 'Arts', 'U.S.'] * 5,
        })
        data_column = 'abstract'
        label_column = 'section'

        train_features, test_features, _, label_list = train_and_test_features_from_df(
                data, data_column, label_column, self.trainer.bert_model_hub,
                self.trainer.max_seq_length)
        self.trainer.train(train_features, label_list)

    def test_train(self):
        self.train_model()

    def test_train_and_test(self):
        self.train_model()
        results = self.trainer.test(test_features)
        print('Evaluation results:', results)
        results2 = self.trainer.test(test_features)
        print('Evaluation results:', results2)
        eval_acc1, eval_acc2 = results['eval_accuracy'], results2['eval_accuracy']
        self.assertEqual(eval_acc1, eval_acc2)
        loss1, loss2 = results['loss'], results2['loss']
        self.assertEqual(eval_acc1, eval_acc2)
        shutil.rmtree(self.output_dir)

    def test_train_and_predict(self):
        self.train_model()
        input_sentences = [
            "test four",
            "test one",
        ] * 5
        preds = self.trainer.predict(input_sentences)

if __name__ == '__main__':
    unittest.main()
