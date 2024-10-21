import numpy as np
import csv
import matplotlib.pyplot as plt

# MORE DATA
# 84000,6200
# 82029,6390
# 63060,6390
# 74000,6600
# 97500,6800
# 67000,6800
# 76025,6900
# 48235,6900
# 93000,6990
# 60949,7490
# 65674,7555
# 54000,7990
# 68500,7990
# 22899,7990
# 61789,8290

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

		self.fig, self.axis = plt.subplots(1, 1)

		self.fig.canvas.manager.set_window_title("Linear Regression")

		try:
			with open(file) as csvfile: #Dimensions = 24 * 2
				reader = csv.DictReader(csvfile)
				self.data = np.array([(int(row['km']), int(row['price'])) for row in reader])
		except:
			raise DataError(f"Failed to initialize data from provide file '{file}'.")

		# self
		self.x_data = self.data[:,0].reshape(-1, 1)
		self.y_data = self.data[:,1].reshape(-1, 1)

		print(self.x_data.shape[0])
		print(self.x_data.shape)
		self.X = np.hstack((self.x_data, np.ones((self.x_data.shape[0], 1))))

		# print(self.X)

		# print(self.data)
		# print(self.data.shape)
		# for x, y in zip(self.x, self.y):
		# 	print(f"f({x}) = {y}")

		self.__plot_data()


	def __plot_data(self):
		# font_axis = {'family':'poppins','color':'#000494','size':12}
		font_axis = {'color':'#000494','size':12}
		# self.axis[0, 0].scatter(self.data[:,0], self.data[:,1])
		plt.scatter(self.x_data, self.y_data)
		plt.xlabel("Mileage (km)", fontdict=font_axis)
		plt.ylabel("Price (€)", fontdict=font_axis)

		# plt.tick_params(axis='x', which='major', labelsize=9)
		# plt.xticks(self.x_data, [str(i) for i in self.x_data], rotation=70)

		plt.tight_layout()
		# self.axis[0, 0].set_title("Data")


	def show(self):
		plt.show()

if __name__ == "__main__":
	try:
		linearRegression = LinearRegression("data.csv")
		linearRegression.show()
	except Exception as e:
		print("Not able to perform linear regression. :")
		print(e)
