import numpy as np

class Normalization:
	"""
	X_standardize = (X - μ)/e

	X: Matrice of value
	μ: Mean of the variable's values
	e: Standard deviation of the variable's values
	"""

	def __init__(self, X: np.ndarray):
		self.X: np.ndarray = X
		self.mean = X.mean()
		self.standard_deviation = X.std()
		print(self.mean)
		print(self.standard_deviation)

	def standardize_all(self):

		self.X = (self.X - self.mean) / self.standard_deviation
		return self.X

	def destandardize_all(self):
		self.X = self.X * self.standard_deviation + self.mean
		return self.X #.astype(int)

	def standardize(self, value) -> float:
		return (value - self.mean) / self.standard_deviation

	def destandardize(self, value) -> float:
		return value * self.standard_deviation + self.mean
