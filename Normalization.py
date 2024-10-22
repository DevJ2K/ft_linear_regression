import numpy as np

class Normalization:
	"""
	X_standardize = (X - μ)/e

	X: Matrice of value
	μ: Mean of the variable's values
	e: Standard deviation of the variable's values
	"""

	def __init__(self, array: np.ndarray):
		self.array: np.ndarray = array
		self.mean = array.mean()
		self.standard_deviation = array.std()

	def standardize_all(self):

		self.array = (self.array - self.mean) / self.standard_deviation
		return self.array

	def destandardize_all(self):
		self.array = self.array * self.standard_deviation + self.mean
		return self.array #.astype(int)

	def standardize(self, value) -> float:
		return (value - self.mean) / self.standard_deviation

	def destandardize(self, value) -> float:
		return value * self.standard_deviation + self.mean

if __name__ == "__main__":
	array = np.array([[1], [2], [3], [4], [5]]).reshape(-1, 1)
	print(array)
	print(array.shape)
	array_normalization = Normalization(array)
	print(array_normalization.standardize_all())
	print(array_normalization.destandardize_all())
