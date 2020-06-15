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
        self.data = pd.DataFrame({
            'abstract': ['test one', 'test two', 'test three'] * 5,
            'section': ['U.S.', 'Arts', 'U.S.'] * 5,
        })

    def train_model(self):
        self.trainer.train(self.data['abstract'], self.data['section'])

    def test_train(self):
        self.train_model()
        shutil.rmtree(self.output_dir)

    def test_train_and_test(self):
        self.train_model()
        results = self.trainer.evaluate(self.data['abstract'], self.data['section'])
        results2 = self.trainer.evaluate(self.data['abstract'], self.data['section'])
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
        shutil.rmtree(self.output_dir)

if __name__ == '__main__':
    unittest.main()
