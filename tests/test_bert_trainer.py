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
