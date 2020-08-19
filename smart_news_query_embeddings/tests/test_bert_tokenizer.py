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
import bert
from smart_news_query_embeddings.preprocessing.bert_tokenizer import \
create_tokenizer, tokenize_data, get_filtered_nyt_data_with_scores

class TestBertTokenizer(unittest.TestCase):

	BERT_DIR = 'uncased_L-12_H-768_A-12'
	MAX_SEQ_LENGTH = 128
	NUM_CLASSES = 64

	def test_create_tokenizer(self):

		"""
		Test that creating the tokenizer does what we expect.
		"""

		tokenizer = create_tokenizer(self.BERT_DIR)
		self.assertIsInstance(tokenizer, bert.tokenization.bert_tokenization.FullTokenizer)

	def test_tokenize_data(self):

		"""
		Test that the tokenizer can create padded input sequences
		and one-hot encode labels correctly.
		"""

		tokenizer = create_tokenizer(self.BERT_DIR)
		test_sentences = ["Hello", "World"]
		test_labels = [0, 1]
		N = len(test_sentences)
		ids, labels = tokenize_data(test_sentences, test_labels, tokenizer,
			self.MAX_SEQ_LENGTH, self.NUM_CLASSES)
		self.assertEqual(ids.shape, (N, self.MAX_SEQ_LENGTH))
		self.assertEqual(labels.shape, (N, self.NUM_CLASSES))

	def test_get_filtered_data(self):

		"""
		Test that getting the filtered NYT data with specificity scores
		has the articles sorted by their specificity score in ascending order.
		"""

		data = get_filtered_nyt_data_with_scores('data/nyt_articles_with_normalized_scores.pkl')
		self.assertTrue((data['normalized_abstract_score'].diff()[1:] >= 0).all())


if __name__ == '__main__':
	unittest.main()