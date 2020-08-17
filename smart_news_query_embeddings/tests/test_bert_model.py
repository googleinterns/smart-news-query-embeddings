import unittest
import bert
import numpy as np
import tensorflow as tf
from smart_news_query_embeddings.models.bert_keras_model import BertKerasModel

class TestBertKerasModel(unittest.TestCase):

	NUM_CLASSES = 63
	BERT_DIR = 'uncased_L-12_H-768_A-12'
	MAX_SEQ_LENGTH = 128
	DENSE_SIZE = 256
	DROPOUT_RATE = 0.5
	NUM_DENSE_LAYERS = 2

	def get_model(self):
		return BertKerasModel(
			num_classes=self.NUM_CLASSES,
			bert_dir=self.BERT_DIR,
			max_seq_length=self.MAX_SEQ_LENGTH,
			dense_size=self.DENSE_SIZE,
			dropout_rate=self.DROPOUT_RATE,
			num_dense_layers=self.NUM_DENSE_LAYERS
		)

	def test_model_init(self):

		"""
		Test that the __init__ call sets all the instance
		attributes properly.
		"""

		model = self.get_model()
		self.assertEqual(model.num_classes, self.NUM_CLASSES)
		self.assertEqual(model.bert_dir, self.BERT_DIR)
		self.assertEqual(model.max_seq_length, self.MAX_SEQ_LENGTH)
		self.assertEqual(model.dense_size, self.DENSE_SIZE)
		self.assertEqual(model.dropout_rate, self.DROPOUT_RATE)
		self.assertEqual(model.num_dense_layers, self.NUM_DENSE_LAYERS)

	def test_model_layers(self):

		"""
		Test that the layers of the instantiated model are what we
		are expecting.
		"""

		model = self.get_model()
		layers = model.layers
		self.assertIsInstance(layers[0], bert.model.BertModelLayer)
		self.assertIsInstance(layers[1], tf.keras.layers.Flatten)
		for i in range(2, 2 + 3 * self.NUM_DENSE_LAYERS, 3):
			self.assertIsInstance(layers[i], tf.keras.layers.Dense)
			self.assertIsInstance(layers[i + 1], tf.keras.layers.LeakyReLU)
			self.assertIsInstance(layers[i + 2], tf.keras.layers.Dropout)
		self.assertIsInstance(layers[-1], tf.keras.layers.Dense)

	def test_model_call(self):

		"""
		Test that the forward pass of the model returns an output
		with exactly the number of classes we specify.
		"""

		model = self.get_model()
		model.build(input_shape=(None, self.MAX_SEQ_LENGTH))
		x = np.zeros((1, self.MAX_SEQ_LENGTH))
		output = model(x).numpy()
		self.assertEqual(output.shape, (1, self.NUM_CLASSES))

	def test_get_embedding(self):

		"""
		Test that we are able to extract embeddings from an inner dense
		layer with the correct shape.
		"""

		model = self.get_model()
		model.build(input_shape=(None, self.MAX_SEQ_LENGTH))
		x = np.zeros((1, self.MAX_SEQ_LENGTH))
		embedding = model.get_embedding(x).numpy()
		self.assertEqual(embedding.shape, (1, self.DENSE_SIZE))

if __name__ == '__main__':
	unittest.main()