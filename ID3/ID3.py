import numpy as np
import math

eps = 0.001

class ID3:
	def __init__(self, X, y, label = -1):
		self.label = label
		y_pos_count = (y == 1).sum()
		y_neg_count = (y == 0).sum()

		if y_pos_count == 0:
			self.label = 0

		elif y_neg_count == 0:
			self.label = 1

		elif len(X.columns) == 0:
			self.label = 1 if y_pos_count >= y_neg_count else 0

		else:
			self.feature = self.maxIG(X, y)

			X_0 = X[ X[self.feature] == 0].copy()
			X_0.drop(self.feature, 1, inplace=True)
			y_0 = y[X[self.feature] == 0].copy()

			if X_0.shape[0] == 0:
				self.left_ID3 = ID3(X_0, y_0, 1 if y_pos_count >= y_neg_count else 0)
			else:
				self.left_ID3 = ID3(X_0, y_0)

			X_1 = X[X[self.feature] == 1].copy()
			X_1.drop(self.feature, 1, inplace=True)
			y_1 = y[X[self.feature] == 1].copy()

			if X_1.shape[0] == 0:
				self.right_ID3 = ID3(X_1, y_1, 1 if y_pos_count >= y_neg_count else 0)
			else:
				self.right_ID3 = ID3(X_1, y_1)


	def calcEntropy(self, X, y):

		p_0 = (y == 0).sum() / (len(X.shape) + eps)
		p_1 = (y == 1).sum() / (len(X.shape) + eps)
		entropy_0 = p_0 * math.log(p_0 + eps)
		entropy_1 = p_1 * math.log(p_1 + eps)

		return -(entropy_0 + entropy_1)


	def calcIG(self, X, y, feature):
		return self.calcEntropy(X, y) - self.calcEntropy(X[X[feature] == 0], y[X[feature] == 0]) \
									- self.calcEntropy(X[X[feature] == 1], y[X[feature] == 1])

	def maxIG(self, X, y):
		IGs = []

		for feature in X.columns:
			IGs.append(self.calcIG(X, y, feature))
		IGs = np.array(IGs)


		return X.columns[np.argmax(IGs)]


	def predict(self, x):

		if self.label != -1:
			return self.label
		else:
			return self.right_ID3.predict(x) if x[self.feature] else self.left_ID3.predict(x)

	def predict_mul(self, X):

		y = []
		for index, row in X.iterrows():
			y.append(self.predict(row))

		return np.array(y)

import pandas as pd

if __name__ == "__main__":
	tranformed_data = pd.read_csv("pairs.csv")
	X = tranformed_data
	Y = tranformed_data['order']
	X.drop('order', 1, inplace=True)
	X = X.values
	Y = Y.values

	from sklearn.utils import shuffle

	X = shuffle(X)
	Y = shuffle(Y)

	root = ID3(tranformed_data, Y)

	y_predicted = root.predict_mul(tranformed_data)

	print(((y_predicted - Y))*((y_predicted - Y)).mean())
