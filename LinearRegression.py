import numpy as np
import csv
import matplotlib.pyplot as plt
from Normalization import Normalization

class DataError(Exception):
	pass

class LinearRegression:
	# figure, axis = plt.subplots(2, 2)

	# # For Sine Function
	# axis[0, 0].plot(X, Y1)
	# axis[0, 0].set_title("Sine Function")

	# # For Cosine Function
	# axis[0, 1].plot(X, Y2)
	# axis[0, 1].set_title("Cosine Function")

	# # For Tangent Function
	# axis[1, 0].plot(X, Y3)
	# axis[1, 0].set_title("Tangent Function")

	# # For Tanh Function
	# axis[1, 1].plot(X, Y4)
	# axis[1, 1].set_title("Tanh Function")

	def __init__(self, file: str):

		# np.set_printoptions(suppress=True)
		# Matplotlib
		# self.fig, self.axis = plt.subplots(2, 2)
		self.fig, self.axis = plt.subplots(3, 1)
		self.fig.canvas.manager.set_window_title("Linear Regression")

		self.cost_history = []

		# Init Data
		try:
			with open(file) as csvfile: #Dimensions = 24 * 2
				reader = csv.DictReader(csvfile)
				self.data = np.array([(int(row['km']), int(row['price'])) for row in reader])
		except:
			raise DataError(f"Failed to initialize data from provide file '{file}'.")

		self.x_data = self.data[:,0].reshape(-1, 1)
		self.y_data = self.data[:,1].reshape(-1, 1)

		normalize_x_data = Normalization(self.x_data)
		normalize_y_data = Normalization(self.y_data)
		# self.x_data = normalize_x_data.standardize_all()
		# self.y_data = normalize_y_data.standardize_all()

		self.x_standardized = normalize_x_data.standardize_all()
		# self.y_standardized = normalize_y_data.standardize_all()
		self.X = np.hstack((self.x_standardized, np.ones((self.x_data.shape[0], 1))))

		# self.X = np.hstack((self.x_data, np.ones((self.x_data.shape[0], 1))))

		# self.theta = np.array([-0.025, 9000]).reshape(2, 1)
		self.theta = np.zeros((2, 1))

		# print(self.data)
		# print(self.data.shape)
		# for x, y in zip(self.x, self.y):
		# 	print(f"f({x}) = {y}")

		# self.__plot_data()


	def __model(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
		"""_summary_

		Returns:
			np.ndarray: Shape (m x 1)
		"""
		# theta = np.array([-0.004, 10.]).reshape(2, 1)
		# np.set_printoptions(suppress=True)
		# print(X)
		# print(y)
		# print(theta)
		# print("X dot Theta")
		# for y_real, y_pred in zip(y, X.dot(theta)):
		# 	print(f"{y_real} - {y_pred}")
		# # print(X.dot(theta))
		# print(X.dot(theta).shape)

		return X.dot(theta)

	def __cost_function(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
		# theta = np.array([-0.004, 10.]).reshape(2, 1)
		m = X.shape[0]
		# print("MODEL PRED :")
		# print(self.__model(X, theta))
		# print("Y")
		# print(y)
		# print("DIFF")
		# print(np.sum((self.__model(X, theta) - y) ** 2))
		# print(1/(2 * m) * np.sum((self.__model(X, theta) - y) ** 2))
		return 1/(2 * m) * np.sum((self.__model(X, theta) - y) ** 2)

	def __gradient(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
		"""_summary_

		Returns:
			np.ndarray: Shape (2 x 1)
		"""
		m = X.shape[0]
		return (1/m) * X.T.dot(self.__model(X, theta) - y)

	def __gradient_descent(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, n_iteration: int, learning_rate: float):
		self.cost_history = np.zeros(n_iteration)
		self.coef_determination_history = np.zeros(n_iteration)

		for i in range(n_iteration):
			theta = theta - learning_rate * self.__gradient(X, y, theta)
			self.cost_history[i] = self.__cost_function(X, y, theta)
			self.coef_determination_history[i] = self.__coef_determination(y, self.__model(X, theta))
		self.theta = theta


	def __coef_determination(self, y: np.ndarray, pred: np.ndarray):
		u = ((y - pred) ** 2).sum()
		v = ((y - y.mean()) ** 2).sum()
		return (1 - u/v)


	def train_model(self, iterations: int = 1000, learningRate: float = 0.005):
		self.__gradient_descent(self.X, self.y_data, self.theta, iterations, learningRate)
		self.__plot_data(iterations=iterations)

		print(self.__coef_determination(self.y_data, self.__model(self.X, self.theta)))
		plt.show()

	def use_model(self, mileage: int) -> float:
		# LOAD LE MODEL
		pass
		# if (mileage < 0):
		# 	raise DataError("Mileage cannot be less than 0.")


	def __plot_data(self, iterations: int):
		# font_axis = {'family':'poppins','color':'#000494','size':12}
		font_axis = {'color':'#000494','size':12}

		(axis_model, axis_cost, axis_precision) = self.axis


		axis_cost.plot(range(iterations), self.cost_history)
		# axis_cost.set_xlabel("Mileage (km)", fontdict=font_axis)
		# axis_cost.set_ylabel("Price (€)", fontdict=font_axis)

		axis_model.scatter(self.x_data, self.y_data)
		axis_model.plot(self.x_data, self.__model(self.X, self.theta), c='r')
		axis_model.set_xlabel("Mileage (km)", fontdict=font_axis)
		axis_model.set_ylabel("Price (€)", fontdict=font_axis)


		axis_precision.plot(range(iterations), self.coef_determination_history)
		# plt.scatter(self.x_data, )
		# default_theta = np.zeros((2, 1))
		# axis_model.plot(self.x_data, self.__model(self.X, default_theta), c='y')

		# self.axis[1].legend([None, 'First List', 'Second List'], loc='upper left')

		# plt.plot(self.x_data, self.__model(self.X, self.y_data, self.theta), c='r')
		# plt.tick_params(axis='x', which='major', labelsize=9)
		# plt.xticks(self.x_data, [str(i) for i in self.x_data], rotation=70)

		plt.tight_layout()
		# self.axis[0, 0].set_title("Data")


	# def __show(self):
	# 	self.__plot_data()
	# 	# print(self.__coef_determination(self.y_data, self.__model(self.X, self.theta)))
	# 	# print(self.theta)
	# 	plt.show()

if __name__ == "__main__":
	try:
		linearRegression = LinearRegression("data.csv")
		linearRegression.train_model(iterations=1500, learningRate=0.005)
		# linearRegression.show()
	except Exception as e:
		print("Not able to perform linear regression. :")
		print(e)
