import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


class MLModel:
	def __init__(self):
		self.model = self.train_model()


	def predict(self, x):
		print(f"predicting {x}")

		input_float = float(x)
		prediction = self.model.predict(np.array([[input_float]]))[0,0]

		return str(prediction)


	def train_model(self):
		# train the mode with dummy data for testing

		X = np.array([[1],[2],[3],[4],[5],[6],[7]])
		y = X * 2 + 3

		return LinearRegression().fit(X, y)

