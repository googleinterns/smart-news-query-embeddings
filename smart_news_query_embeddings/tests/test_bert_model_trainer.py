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

import os
import shutil
import uuid
import unittest
from smart_news_query_embeddings.trainers.bert_model_trainer import BertModelTrainer

class TestBertModelTrainer(unittest.TestCase):

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-5
    MAX_SEQ_LENGTH = 128
    DROPOUT_RATE = 0.5
    EPOCHS = 1
    DENSE_SIZE = 256
    BERT_DIR = 'uncased_L-12_H-768_A-12'

    def tear_down(self):
        exp_dir = os.path.join('experiments', self.exp_id)
        shutil.rmtree(exp_dir)

    def get_model(self):
        self.exp_id = str(uuid.uuid4())
        return BertModelTrainer(self.exp_id, batch_size=self.BATCH_SIZE, learning_rate=self.LEARNING_RATE,
        max_seq_length=self.MAX_SEQ_LENGTH, dropout_rate=self.DROPOUT_RATE, epochs=self.EPOCHS,
        dense_size=self.DENSE_SIZE, bert_dir=self.BERT_DIR, dry_run=True)

    def test_init(self):

        """
        Test that the trainer properly initializes all variables.
        """

        trainer = self.get_model()
        self.assertEqual(trainer.exp_dir, os.path.join('experiments', self.exp_id))
        self.assertEqual(trainer.batch_size, self.BATCH_SIZE)
        self.assertEqual(trainer.learning_rate, self.LEARNING_RATE)
        self.assertEqual(trainer.max_seq_length, self.MAX_SEQ_LENGTH)
        self.assertEqual(trainer.dense_size, self.DENSE_SIZE)
        self.assertEqual(trainer.bert_dir, self.BERT_DIR)
        self.assertEqual(trainer.train_x.shape[0], 15)
        self.assertEqual(trainer.valid_x.shape[0], 5)
        self.assertEqual(trainer.train_y.shape[0], 15)
        self.assertEqual(trainer.valid_y.shape[0], 5)
        self.tear_down()

    def test_save_after_train(self):

        """
        Test that the after training, the directory containing
        all the metrics and data for the experiment has all the
        subfolders we expect.
        """

        trainer = self.get_model()
        trainer.train()
        exp_dir = os.path.join('experiments', self.exp_id)
        data_dir = os.path.join(exp_dir, 'data')
        embeddings_dir = os.path.join(exp_dir, 'embeddings')
        model_dir = os.path.join(exp_dir, 'model')
        self.assertTrue(os.path.exists(exp_dir))
        self.assertTrue(os.path.exists(data_dir))
        self.assertTrue(os.path.exists(model_dir))
        self.tear_down()


if __name__ == '__main__':
    unittest.main()