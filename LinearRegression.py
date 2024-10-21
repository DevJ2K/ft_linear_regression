import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

		# Matplotlib
		# self.fig, self.axis = plt.subplots(2, 2)
		self.fig, self.axis = plt.subplots(2, 1)
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


		scaler_x = StandardScaler()
		scaler_y = StandardScaler()

		# self.x_data = scaler_x.fit_transform(self.x_data)
		# self.y_data = scaler_y.fit_transform(self.y_data)

		self.X = np.hstack((scaler_x.fit_transform(self.x_data), np.ones((self.x_data.shape[0], 1))))

		# PERFECT THETA MANO
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
		# np.set_printoptions(suppress=True)
		# print(X.T)
		# print(X.T.shape)
		# print(self.__model(self.X, self.theta).shape)
		# print(((1/m) * X.T.dot(self.__model(self.X, self.theta) - y)) * 0.0000005 )


		# print("Theta")
		# print(self.theta)
		# print("X")
		# print(self.X)
		# print("Model")
		# print(self.__model(self.X, self.theta))
		# print("Model - Y")
		# print(self.__model(X, theta) - y)
		# print((1/m) * np.sum(self.__model(X, theta) - y))
		# np.set_printoptions(suppress=True)

		# print(X.T)
		# print((1/m) * X.T.dot(self.__model(X, theta) - y))


		return (1/m) * X.T.dot(self.__model(X, theta) - y)

	def __gradient_descent(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray, n_iteration: int, learning_rate: float):
		self.cost_history = np.zeros(n_iteration)

		self.__gradient(X, y, theta)
		# np.set_printoptions(suppress=True)
		# np.set_printoptions(suppress=True, precision=2, floatmode='fixed')
		# print(learning_rate * self.__gradient(X, y, theta))
		for i in range(n_iteration):
			theta = theta - learning_rate * self.__gradient(X, y, theta)
			self.cost_history[i] = self.__cost_function(X, y, theta)
		self.theta = theta
		# self.theta = np.array([-0.025, 9000]).reshape(2, 1)


	def train_model(self):
		# self.__model(self.X, self.theta)
		# self.__cost_function(self.X, self.y_data, self.theta)
		# self.__gradient(self.X, self.y_data, self.theta)
		np.set_printoptions(suppress=True)
		self.__gradient_descent(self.X, self.y_data, self.theta, 1000, 0.01)

		# print("Theta")
		# print(self.theta)
		# print("X")
		# print(self.X)
		# print("Model")
		# print(self.__model(self.X, self.theta))
		# print("Model - Y")
		# print(self.y_data)
		pass

	def use_model(self, mileage: int) -> float:
		pass

	def __plot_data(self):
		# font_axis = {'family':'poppins','color':'#000494','size':12}
		font_axis = {'color':'#000494','size':12}
		# self.axis[0, 0].scatter(self.data[:,0], self.data[:,1])
		self.axis[0].scatter(self.x_data, self.y_data)
		self.axis[0].set_xlabel("Mileage (km)", fontdict=font_axis)
		self.axis[0].set_ylabel("Price (€)", fontdict=font_axis)

		self.axis[1].scatter(self.x_data, self.y_data)
		self.axis[1].set_xlabel("Mileage (km)", fontdict=font_axis)
		self.axis[1].set_ylabel("Price (€)", fontdict=font_axis)

		# default_theta = np.zeros((2, 1))
		# plt.plot(self.X, self.__model(self.X, default_theta), c='y')
		self.axis[1].plot(self.x_data, self.__model(self.X, self.theta), c='r')

		# plt.plot(self.x_data, self.__model(self.X, self.y_data, self.theta), c='r')
		# plt.tick_params(axis='x', which='major', labelsize=9)
		# plt.xticks(self.x_data, [str(i) for i in self.x_data], rotation=70)

		# plt.tight_layout()
		# self.axis[0, 0].set_title("Data")


	def show(self):
		self.__plot_data()
		plt.show()

if __name__ == "__main__":
	try:
		linearRegression = LinearRegression("data.csv")
		linearRegression.train_model()
		linearRegression.show()
	except Exception as e:
		print("Not able to perform linear regression. :")
		print(e)
