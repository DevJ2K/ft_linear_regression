import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Normalization import Normalization

class DataError(Exception):
	pass

class LinearRegression:

	def __init__(self):
		self.plot_info = {
			'window_title': 'Linear Regression',
			'window_bg': '#d1d2ff',
			'data_color': '#0c0066',
			'model_color': '#eb7eff',
			'font_axis': {'color':'#252525'},
			'font_title': {'color':'#000000'}
		}


	def __model(self, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
		"""_summary_

		Returns:
			np.ndarray: Shape (m x 1)
		"""
		return X.dot(theta)

	def __cost_function(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
		m = X.shape[0]
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


	def update(self, i):
		plot_info = self.plot_info

		self.theta = self.theta - self.learning_rate * self.__gradient(self.X, self.y_standardized, self.theta)
		self.cost_history[i] = self.__cost_function(self.X, self.y_standardized, self.theta)
		self.coef_determination_history[i] = self.__coef_determination(self.y_standardized, self.__model(self.X, self.theta))

		self.axis_model.clear()

		self.axis_model.scatter(self.x_data, self.y_data, c=plot_info['data_color'])
		self.axis_model.get_ylim()
		# self.axis_model.set_ylim([self.y_data.min(), self.y_data.max()])
		self.axis_model.set_ylim(self.axis_model.get_ylim())

		self.axis_model.plot(self.x_data, self.normalize_y_data.destandardize(self.__model(self.X, self.theta)), c=plot_info['model_color'])

		self.axis_model.set_xlabel("Mileage (km)", fontdict=plot_info['font_axis'])
		self.axis_model.set_ylabel("Price (€)", fontdict=plot_info['font_axis'])
		self.axis_model.legend(['Training Data','Model Prediction'], loc='upper right')
		self.axis_model.set_title("Model Prediction vs Training Data", fontdict=plot_info['font_title'])



	def __animate_gradient_descent(self, n_iteration: int):
		self.fig, self.axis_model = plt.subplots(1, 1)
		self.fig.canvas.manager.set_window_title(self.plot_info['window_title'])
		self.fig.set_facecolor(self.plot_info['window_bg'])

		self.cost_history = np.zeros(n_iteration)
		self.coef_determination_history = np.zeros(n_iteration)

		anim = FuncAnimation(self.fig, self.update, frames=np.arange(1, n_iteration), interval=0)
		plt.show()

	def __coef_determination(self, y: np.ndarray, pred: np.ndarray):
		"""
		To calculate the precision of the model.
		"""
		u = ((y - pred) ** 2).sum()
		v = ((y - y.mean()) ** 2).sum()
		return (1 - u/v)


	def train_model(self,
				file_info: dict,
				iterations: int = 1000,
			learningRate: float = 0.01,
			animate: bool = False
			):

		try:
			file = file_info['filename']
			x_type = file_info['x_type']
			y_type = file_info['y_type']
			x_name = file_info['x_name']
			y_name = file_info['y_name']
			file = file_info['filename']
			file = file_info['filename']
			with open(file) as csvfile:
				reader = csv.DictReader(csvfile)
				self.data = np.array([(x_type(row[x_name]), y_type(row[y_name])) for row in reader])
		except:
			raise DataError(f"Failed to initialize data from provide file '{file}'.")

		self.learning_rate = learningRate

		# We don't standardise this data.
		self.x_data = self.data[:,0].reshape(-1, 1)
		self.y_data = self.data[:,1].reshape(-1, 1)


		self.normalize_x_data = Normalization(self.x_data)
		self.normalize_y_data = Normalization(self.y_data)

		# Data standardized
		self.x_standardized = self.normalize_x_data.standardize_all()
		self.y_standardized = self.normalize_y_data.standardize_all()


		# We use a standardized X matrice to train the model.
		self.X = np.hstack((self.x_standardized, np.ones((self.x_data.shape[0], 1))))
		self.theta = np.zeros((2, 1))


		# print(self.__coef_determination(self.y_data, self.__model(self.X, self.theta)))
		if (animate == True):
			self.__animate_gradient_descent(iterations)
		else:
			self.__gradient_descent(self.X, self.y_standardized, self.theta, iterations, learningRate)
			self.__plot_data(iterations=iterations)

	def use_model(self, mileage) -> float:
		if (mileage < 0):
			raise DataError("Mileage cannot be less than 0.")
		print("PREDICTION")
		standardise_mileage = self.normalize_x_data.standardize(mileage)
		print(standardise_mileage)
		array_mileage = np.hstack(([[standardise_mileage]], np.ones((1, 1))))
		print(array_mileage)
		print(array_mileage.shape)
		print(self.theta.shape)

		# print(array_mileage.dot(self.theta))
		print(self.normalize_y_data.destandardize(array_mileage.dot(self.theta)))
		# pred = self.theta[0] * mileage + self.theta[1]
		# print(pred)
		# print(self.normalize_y_data.destandardize(pred))
		# print(self.__model([240000], self.theta))
		# LOAD LE MODEL



	def __plot_data(self, iterations: int):

		plot_info = self.plot_info
		self.fig, self.axis = plt.subplots(2, 2)
		self.fig.canvas.manager.set_window_title(plot_info['window_title'])

		self.axis_model = self.axis[0][0]
		self.axis_diff_cost_precision = self.axis[0][1]
		self.axis_cost = self.axis[1][0]
		self.axis_precision = self.axis[1][1]


		# https://matplotlib.org/stable/users/explain/animations/animations.html
		# Implement curses animation

		# self.fig.canvas.manager.set_window_title("Linear Regression")

		font_axis = {'color':'#252525'}#,'size':12}
		font_title = {'color':'#000000'}#,'size':12}

		# (axis_model, axis_diff_cost_precision, axis_cost, axis_precision) = self.axis
		# axis_model = self.axis[0][0]
		# axis_diff_cost_precision = self.axis[0][1]
		# axis_cost = self.axis[1][0]
		# axis_precision = self.axis[1][1]

		self.fig.set_facecolor(plot_info['window_bg'])

		self.axis_model.scatter( self.x_data, self.y_data, c=plot_info['data_color'])

		self.axis_model.plot(self.x_data, self.normalize_y_data.destandardize(self.__model(self.X, self.theta)), c=plot_info['model_color'])
		self.axis_model.set_xlabel("Mileage (km)", fontdict=plot_info['font_axis'])
		self.axis_model.set_ylabel("Price (€)", fontdict=plot_info['font_axis'])
		self.axis_model.legend(['Training Data','Model Prediction'], loc='upper right')
		self.axis_model.set_title("Model Prediction vs Training Data", fontdict=plot_info['font_title'])



		self.axis_diff_cost_precision.plot(range(iterations), self.coef_determination_history, c='g')
		self.axis_diff_cost_precision.plot(range(iterations), self.cost_history, c='r')
		self.axis_diff_cost_precision.set_yticks([(i) for i in np.arange(0, 1.25, 0.25)])
		self.axis_diff_cost_precision.legend(['Precision (R²)','Cost Function'], loc='upper right')
		self.axis_diff_cost_precision.set_xlabel("Iterations", fontdict=plot_info['font_axis'])
		self.axis_diff_cost_precision.axes.get_yaxis().set_visible(False)
		self.axis_diff_cost_precision.set_title("Cost Function vs Precision Over Iterations", fontdict=plot_info['font_title'])

		self.axis_cost.plot(range(iterations), self.cost_history, c='r')
		self.axis_cost.legend(['Cost Function'], loc='upper right')
		self.axis_cost.set_xlabel("Iterations", fontdict=plot_info['font_axis'])
		self.axis_cost.set_ylabel("Cost", fontdict=plot_info['font_axis'])
		self.axis_cost.set_title("Cost Function Evolution Over Iterations", fontdict=plot_info['font_title'])

		self.axis_precision.plot(range(iterations), self.coef_determination_history * 100, c='g')
		self.axis_precision.set_yticks([i for i in np.arange(0, 125, 25)])
		self.axis_precision.legend(['Precision (R² in %)'], loc='upper right')
		self.axis_precision.set_ylabel("Precision (%)", fontdict=plot_info['font_axis'])
		self.axis_precision.set_xlabel("Iterations", fontdict=plot_info['font_axis'])
		self.axis_precision.set_title("Model Precision Evolution (R²) Over Iterations", fontdict=plot_info['font_title'])

		plt.tight_layout()
		plt.show()

if __name__ == "__main__":
	try:
		file_info = {
			'filename': 'data.csv',
			'x_name': 'km',
			'x_type': int,
			'y_name': 'price',
			'y_type': int,
		}
		# file_info = {
		# 	'filename': 'perfect_data.csv',
		# 	'x_name': 'x',
		# 	'x_type': int,
		# 	'y_name': 'y',
		# 	'y_type': float,
		# }
		linearRegression = LinearRegression()
		linearRegression.train_model(file_info=file_info, iterations=1000, learningRate=0.01, animate=False)
		# linearRegression.use_model(int(input("Entrez votre kilometrage")))
		# linearRegression.show()
	except Exception as e:
		print("Not able to perform linear regression. :")
		print(e)
