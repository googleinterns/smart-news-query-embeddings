import numpy as np
from smart_news_query_embeddings.models.two_tower_model import TwoTowerModel
from tensorflow.keras.optimizers import Adam
from bert_model_trainer import BertModelTrainer

class BertModelSpecificityScoreTrainer(BertModelTrainer):

	def __init__(self, exp_name, max_seq_length=128, bert_dir='uncased_L-12_H-768_A-12',
        dense_size=256, learning_rate=1e-5, dropout_rate=0.5, epochs=3, batch_size=32,
        tail_cutoff=0.5, train_tail=False):
		super().__init__(exp_name, max_seq_length, bert_dir, dense_size, learning_rate, dropout_rate
			epochs, batch_size)
		self.tail_cutoff = tail_cutoff
		self.train_tail = train_tail

	def get_train_and_valid_split(self, df):
		CUTOFF = int(self.tail_cutoff * df.shape[0])
		train_df, test_df = df[-CUTOFF:], df[:CUTOFF]
		if self.train_tail:
			train_df, test_df = test_df, train_df
		return train_df, test_df