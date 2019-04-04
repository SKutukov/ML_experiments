import numpy as np
import math

class ID3:
	def __init__(self, X, y, label = -1):
		self.label = label
		y_pos_count = y[y == 1].shape[0]
		y_neg_count = y[y == 0].shape[0]

		if y_pos_count == 0:
			self.label = 1

		elif y_neg_count == 0:
			self.label = 0

		elif len(X.columns) == 0:
			self.label = 1 if y_pos_count >= y_neg_count else 0

		self.feature = self.maxIG(X, y)

		X_0 = X[ X[self.feature] == 0]
		X_0.drop(self.feature, 1, inplace=True)
		y_0 = y[ X[self.feature] == 0]
		if X_0.shape[0] == 0:
			self.left_ID3 = ID3(X_0, y_0, 1 if y_pos_count >= y_neg_count else 0)
		else:
			self.left_ID3 = ID3(X_0, y_0)

		X_1 = X[X[self.feature] == 1]
		X_1.drop(self.feature, 1, inplace=True)
		y_1 = y[X[self.feature] == 1]
		if X_1.shape[0] == 0:
			self.right_ID3 = ID3(X_1, y_1, 1 if y_pos_count >= y_neg_count else 0)
		else:
			self.left_ID3 = ID3(X_1, y_1)

	def calcEntropy(self, X, y):
		p_0 = sum(y == 0).shape[0] / X.shape[0]
		p_1 = sum(y == 1) / X.shape[0]
		entropy_0 = p_0 * math.log(p_0)
		entropy_1 = p_1 * math.log(p_1)

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

