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
import numpy as np
from smart_news_query_embeddings.preprocessing.specificity_scores import \
get_specificity_scores, get_token_scores, get_normalized_scores

class TestSpecificityScores(unittest.TestCase):

	def test_get_specificity_scores(self):

		"""
		Test that we get back as many scores as we put in sentences.
		"""

		sentences = [
			"This is sentence 1 about Wall Street",
			"This is sentence 2 about Donald Trump",
			"This is sentence 3 about New York City",
			"Tesla announces first ever car that can drive in space",
			"A look at last night's game between the Warriors and the Lakers, sponsored by Tesla"
		]
		scores = get_specificity_scores(sentences)
		self.assertEqual(scores.shape[0], len(sentences))

	def test_get_token_scores(self):

		"""
		Tests the mathematical correctness of the token scoring
		algorithm.
		"""

		token_counts = [2, 4, 5, 3, 2, 2]
		num_types = 2
		tokens = []
		types = []
		type_number = 0
		for i, c in enumerate(token_counts):
			token = 'token{}'.format(i + 1)
			tokens.extend([token] * c)
			token_type = 'type{}'.format(type_number + 1)
			types.extend([token_type] * c)
			type_number = (type_number + 1) % num_types
		token_df = pd.DataFrame({
			'token': tokens,
			'type': types,
		})
		scores = get_token_scores(token_df)
		self.assertEqual(scores.shape[0], len(token_counts))
		sorted_scores = sorted(list(scores['score']))
		sorted_token_counts = sorted(token_counts)
		for score, count in zip(sorted_scores, sorted_token_counts):
			expected_score = np.log(count / 3)
			self.assertTrue(np.abs(score - expected_score) < 1e-5)

	def test_get_normalized_scores(self):

		"""
		Test that the normalized scores for given abstracts are correct.
		This function should average the token scores for every token found
		in the abstract and assign that score to the abstract.
		"""

		token_lists = [
			[['token1', 'type1'], ['token2', 'type1'], ['token1', 'type2']]
		]

		indexed_token_scores = pd.DataFrame({
			'token': ['token1', 'token2', 'token1'],
			'type': ['type1', 'type1', 'type2'],
			'score': [3, 4, 5]
		}).set_index(['type', 'token'])
		scores = get_normalized_scores(token_lists, indexed_token_scores)
		self.assertEqual(scores.iloc[0], 4)
		self.assertEqual(scores.shape[0], 1)

if __name__ == '__main__':
	unittest.main()
