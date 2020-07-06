from two_tower_model import TwoTowerModel
from tensorflow.keras.optimizers import Adam
import numpy as np

if __name__ == '__main__':
	model = TwoTowerModel(num_classes=64)
	model.build(input_shape=[(None, 128), (None, 64)])
	print(model.summary())
	x = np.zeros((1, 128))
	label = np.zeros((1, 64))
	output1 = model([x, label]).numpy()
	x1 = np.copy(x)
	label1 = np.zeros((1, 64))
	output2 = model([x1, label1]).numpy()
	assert (output1 == output2).all()
	print(output1)