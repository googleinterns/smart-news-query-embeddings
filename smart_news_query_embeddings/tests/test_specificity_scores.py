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
get_scores_from_spacy_responses, get_specificity_scores

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

	def test_get_scores_from_spacy_responses(self):

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
		token_lists = [
			[['token1', 'type1']],
			[['token2', 'type2']],
			[['token3', 'type1']],
			[['token4', 'type2']],
			[['token5', 'type1']],
			[['token6', 'type2']],
			[['token1', 'type1'], ['token4', 'type2']]
		]
		scores = get_scores_from_spacy_responses(token_lists, token_df)
		self.assertEqual(scores.shape[0], len(token_lists))

		# The mean token count of each class is 3, because type1 has counts [2, 5, 2]
		# and type2 has counts [4, 3, 2]
		# So, we expect the score of each (token, entity) pair to be the log of the
		# count of that pair divided by 3
		expected_scores = list(np.log(np.array(token_counts) / 3))
		# We expect the last token list to average the token scores of token1 and token4
		expected_scores.append((expected_scores[0] + expected_scores[3]) / 2)
		for score, expected_score in zip(scores, expected_scores):
			self.assertTrue(np.abs(score - expected_score) < 1e-5)

if __name__ == '__main__':
	unittest.main()
