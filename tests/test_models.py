from unittest import TestCase, mock, main
from bert_keras_layer import BertKerasModel

class TestTwoTowerModel(TestCase):

	def test_case(self):
		model = BertKerasModel(64)

if __name__ == '__main__':
	main()