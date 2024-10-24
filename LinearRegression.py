import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Normalization import Normalization
import typer
import time
import json
import os
from Colors import *

class DataError(Exception):
	pass

class LinearRegression:

	def __init__(self):
		ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

		self.MODELS_PATH = os.path.join(ROOT_PATH, "models")
		self.DATAS_PATH = os.path.join(ROOT_PATH, "datas")

		self.plot_info = {
			'window_title': 'Linear Regression',
			'window_bg': '#d1d2ff',
			'data_color': '#0c0066',
			'model_color': '#eb7eff',
			'font_axis': {'color':'#252525'},
			'font_title': {'color':'#000000'}
		}

	def show_informations(self):
		# print(f"{BHWHITE}*{RESET}" * 40)
		print(f"{BHWHITE}** STATISTICS **********************{RESET}")
		print(f"{BHYELLOW}WARNING: Theta was trained on standardized data.{RESET}")

		print(f"{BHGREEN}Thetaθ(0) (Weight): {GREEN}{self.theta[0]}{RESET}")
		print(f"{BHGREEN}Thetaθ(1) (Bias): {GREEN}{self.theta[1]}{RESET}\n")

		print(f"{BHWHITE}** LEARNING INFORMATIONS ***********{RESET}")
		print(f"{BHMAG}Iterations : {MAG}{self.n_iterations}{RESET}")
		print(f"{BHMAG}Learning Rate : {MAG}{self.learning_rate}{RESET}\n")

		print(f"{BHWHITE}** STANDARDIZATION INFORMATIONS ****{RESET}")
		print(f"{BHCYAN}Mean X (μ): {CYAN}{self.normalize_x_data.mean}{RESET}")
		print(f"{BHCYAN}Mean Y (μ): {CYAN}{self.normalize_y_data.mean}{RESET}")
		print(f"{BHBLUE}Standard Deviation X (σ): {BLUE}{self.normalize_x_data.standard_deviation}{RESET}")
		print(f"{BHBLUE}Standard Deviation Y (σ): {BLUE}{self.normalize_y_data.standard_deviation}{RESET}\n")

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

		# with typer.progressbar(range(n_iteration)) as progress:
			# for i in progress:
		for i in range(n_iteration):
			theta = theta - learning_rate * self.__gradient(X, y, theta)
			self.cost_history[i] = self.__cost_function(X, y, theta)
			self.coef_determination_history[i] = self.__coef_determination(y, self.__model(X, theta))
				# time.sleep(0.001)

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

		anim = FuncAnimation(self.fig, self.update, frames=n_iteration, interval=0, repeat=False)
		plt.show()

	def __coef_determination(self, y: np.ndarray, pred: np.ndarray):
		"""
		To calculate the precision of the model.
		"""
		u = ((y - pred) ** 2).sum()
		v = ((y - y.mean()) ** 2).sum()
		return (1 - u/v)

	def __save_model(self):
		try:
			if self.file_info["model_file"] is not None:
				file = str(self.file_info["model_file"])
			else:
				raise Exception
		except:
			file = "model_" + self.file_info["filename"]

		data = {
			"precision": self.coef_determination_history[-1],
			"iteration": self.n_iterations,
			"learning_rate": self.learning_rate,
			"theta": self.theta.tolist(),
			"mean_x": self.normalize_x_data.mean,
			"mean_y": self.normalize_y_data.mean,
			"standard_deviation_x": self.normalize_x_data.standard_deviation,
			"standard_deviation_y": self.normalize_y_data.standard_deviation,
		}
		try:
			file_path = os.path.join(self.MODELS_PATH, file)
			with open(file_path, 'w') as json_file:
				json.dump(data, json_file, indent=4)
			print(f"{BHGREEN}SUCCESS: {GREEN}The training has been successfully saved in the file '{file_path}'.{RESET}\n")
		except:
			print(f"{BHRED}FAILURE: {RED}Failed to save the model in '{file_path}'. Please check file permissions or storage space and try again.{RESET}\n")

	def train_model(self,
			config_file: str,
			iterations: int = 1000,
			learningRate: float = 0.01,
			animate: bool = False
			):
		if (iterations <= 0):
			print("Iterations cannot be minus or equal 0")
			return
		try:
			with open(os.path.join(self.DATAS_PATH, "configs", config_file)) as json_file:
				self.file_info = json.load(json_file)
				self.file_info['filename'] = config_file

			file = self.file_info['datafile']
			x_type = eval(self.file_info['x_type'])
			y_type = eval(self.file_info['y_type'])

			with open(os.path.join(self.DATAS_PATH, "sets", file)) as csvfile:
				reader = csv.DictReader(csvfile)
				self.data = np.array([(x_type(row[reader.fieldnames[0]]), y_type(row[reader.fieldnames[1]])) for row in reader])
		except:
			raise DataError(f"Failed to initialize data from provide file '{config_file}'.")

		self.learning_rate = learningRate
		self.n_iterations = iterations

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
			print(f"{BHYELLOW}WARNING: {YELLOW}The model will be displayed with animation, but the training session will not be saved due to performance considerations.{RESET}")
			self.__animate_gradient_descent(iterations)
		else:
			self.__gradient_descent(self.X, self.y_standardized, self.theta, iterations, learningRate)
			self.__save_model()
			self.show_informations()
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
		config_file = "perfect_data.json"
		linearRegression = LinearRegression()
		linearRegression.train_model(config_file=config_file, iterations=1000, learningRate=0.01, animate=False)
		# linearRegression.use_model(int(input("Entrez votre kilometrage")))
		# linearRegression.show()
	except Exception as e:
		print("Not able to perform linear regression. :")
		print(e)
