from bert_keras_model import BertKerasModel
import numpy as np

if __name__ == '__main__':
	model = BertKerasModel(num_classes=64)
	model.build(input_shape=(None, 128))
	print(model.summary())
	x = np.zeros((1, 128))
	output1 = model.get_embedding(x).numpy()
	x1 = np.copy(x)
	output2 = model.get_embedding(x1).numpy()
	assert (output1 == output2).all()
