import unittest
import tempfile
import shutil
import bert
import numpy as np
import tensorflow as tf
from smart_news_query_embeddings.models.two_tower_model import TwoTowerModel

class TestTwoTowerModel(unittest.TestCase):

    NUM_CLASSES = 63
    BERT_DIR = 'uncased_L-12_H-768_A-12'
    MAX_SEQ_LENGTH = 128
    DENSE_SIZE = 256
    DROPOUT_RATE = 0.5
    NUM_DENSE_LAYERS = 2
    LEARNING_RATE = 1e-2

    def get_model(self):
        return TwoTowerModel(
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
        # BERT Layer
        self.assertIsInstance(layers[0], bert.model.BertModelLayer)
        # Flatten BERT output
        self.assertIsInstance(layers[1], tf.keras.layers.Flatten)
        # 2 dense layers in first tower, each with batch norm
        for i in range(2, 2 + 2 * 4, 4):
            self.assertIsInstance(layers[i], tf.keras.layers.Dense)
            self.assertIsInstance(layers[i + 1], tf.keras.layers.BatchNormalization)
            self.assertIsInstance(layers[i + 2], tf.keras.layers.LeakyReLU)
            self.assertIsInstance(layers[i + 3], tf.keras.layers.Dropout)
        # 2 dense layers in second tower for label
        for i in range(10, 2 + 2 * 3, 3):
            self.assertIsInstance(layers[i], tf.keras.layers.Dense)
            self.assertIsInstance(layers[i + 1], tf.keras.layers.LeakyReLU)
            self.assertIsInstance(layers[i + 2], tf.keras.layers.Dropout)
        # final dense layer and output layer (concatenate only in call method)
        self.assertIsInstance(layers[-3], tf.keras.layers.Dense)
        self.assertIsInstance(layers[-2], tf.keras.layers.LeakyReLU)
        self.assertIsInstance(layers[-1], tf.keras.layers.Dense)

    def test_model_call(self):
        model = self.get_model()
        model.build(input_shape=[(None, self.MAX_SEQ_LENGTH), (None, self.NUM_CLASSES)])
        x = np.zeros((1, self.MAX_SEQ_LENGTH))
        y = np.zeros((1, self.NUM_CLASSES))
        output = model((x, y)).numpy()
        self.assertEqual(output.shape, (1, 2))

    def test_get_embedding(self):

        """
        Test that we are able to extract embeddings from an inner dense
        layer with the correct shape.
        """

        model = self.get_model()
        model.build(input_shape=[(None, self.MAX_SEQ_LENGTH), (None, self.NUM_CLASSES)])
        x = np.zeros((1, self.MAX_SEQ_LENGTH))
        embedding = model.get_embedding(x).numpy()
        self.assertEqual(embedding.shape, (1, self.DENSE_SIZE))

    def test_outputs(self):

        """
        Make sure that the outputs and embeddings are different before and after training.
        """
        model = self.get_model()
        model.build(input_shape=[(None, self.MAX_SEQ_LENGTH), (None, self.NUM_CLASSES)])
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=self.LEARNING_RATE),
            metrics=['accuracy'])
        x = np.zeros((1, self.MAX_SEQ_LENGTH))
        y = np.zeros((1, self.NUM_CLASSES))
        y[0, 20] = 0
        out_before = model.predict((x, y))
        embedding_before = model.get_embedding(x).numpy()
        output = np.array([[0, 1]])
        model.fit((x, y), output, epochs=5)
        out_after = model.predict((x, y))
        embedding_after = model.get_embedding(x).numpy()
        self.assertFalse(np.array_equal(out_before, out_after))
        self.assertFalse(np.array_equal(embedding_before, embedding_after))

    def test_model_save(self):

        """
        Test that we can serialize the model and reload it, and still
        get embeddings from it. This is the primary use case of these
        trained models for this project.
        """

        out_dir = tempfile.mkdtemp()
        model = self.get_model()
        model.build(input_shape=[(None, self.MAX_SEQ_LENGTH), (None, self.NUM_CLASSES)])
        x = np.zeros((1, self.MAX_SEQ_LENGTH))
        y = np.zeros((1, self.NUM_CLASSES))
        output = model.predict((x, y))
        model.save(out_dir)
        model = tf.keras.models.load_model(out_dir)
        embedding = model.get_embedding(x).numpy()
        self.assertEqual(embedding.shape, (1, self.DENSE_SIZE))
        shutil.rmtree(out_dir)


if __name__ == '__main__':
    unittest.main()