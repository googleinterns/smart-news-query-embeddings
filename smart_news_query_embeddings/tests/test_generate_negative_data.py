import unittest
import pandas as pd
from smart_news_query_embeddings.preprocessing.generate_negative_data import \
generate_negatives, generate_negatives_from_spacy_responses
from smart_news_query_embeddings.preprocessing.bert_tokenizer import \
get_filtered_nyt_data_with_scores

class TestGenerateNegativeData(unittest.TestCase):

	DATA_PATH = 'data/nyt_articles_with_normalized_scores.pkl'

	def test_generate_negatives(self):

		"""
		Make sure that we get as many negatives as we're expecting.
		"""

		df = get_filtered_nyt_data_with_scores(self.DATA_PATH).sample(100)
		RATIOS = [0.25, 0.5, 1, 2, 5]
		for ratio in RATIOS:
			negatives = generate_negatives(df['fixed_abstract'], df['section'], ratio=ratio)
			self.assertEqual(negatives.shape[0], int(ratio * len(df)))

	def test_generate_negatives_from_spacy_responses(self):

		"""
		Test that the algorithm generates negatives from sentences with different labels but containing
		the same token types.
		"""

		token_lists = pd.Series([
			[['token1', 'type1']],
			[['token5', 'type1']]
		])

		sentences = pd.Series([
			'sentence1',
			'sentence2',
		])

		labels = pd.Series([
			'label1',
			'label2',
		])
		ratio = 1
		negatives = generate_negatives_from_spacy_responses(token_lists, sentences, labels, ratio=ratio)
		self.assertEqual(negatives.shape[0], 2)

		# the negatives should be ('sentence1', 'label2') and ('sentence2', 'label1')
		self.assertEqual(negatives[negatives['sentence'] == 'sentence1'].iloc[0]['label'], 'label2')
		self.assertEqual(negatives[negatives['sentence'] == 'sentence2'].iloc[0]['label'], 'label1')

if __name__ == '__main__':
	unittest.main()