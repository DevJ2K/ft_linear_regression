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

		# _, self.axis = plt.subplots(2, 2)

		try:
			with open(file) as csvfile: #Dimensions = 24 * 2
				reader = csv.DictReader(csvfile)
				self.data = np.array([(row['km'], row['price']) for row in reader])
		except:
			raise DataError(f"Failed to initialize data from provide file '{file}'.")

		print(self.data)
		self.__plot_data()


	def __plot_data(self):
		# self.axis[0, 0].scatter(self.data[:,0], self.data[:,1])
		plt.scatter(self.data[:,0], self.data[:,1])
		# self.axis[0, 0].set_title("Data")


	def show(self):
		plt.show()

if __name__ == "__main__":
	linearRegression = LinearRegression("data.csv")
	linearRegression.show()
