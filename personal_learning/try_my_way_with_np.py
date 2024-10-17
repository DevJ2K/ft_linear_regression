# IMPORT
import csv
import matplotlib.pyplot as plt
import numpy as np

# y=3.7x+8
with open('perfect_data.csv') as csvfile:
	reader = csv.DictReader(csvfile)
	dataset = [(float(row['x']), float(row['y'])) for row in reader]
	array = np.array(dataset)

# print(f"Matrices :\n{array}")

# 3.7x + 8
theta = np.array([0., 0.]).reshape(2, 1)

x_matrix = array[:, 0].reshape(-1, 1)  # Colonne des x
y_matrix = array[:, 1].reshape(-1, 1)  # Colonne des y


X = np.hstack((x_matrix, np.ones(x_matrix.shape)))  # to use linear function with a bias
print("X:")
print(X[5:15])
# print(X.shape)
# print(theta)
# print(theta.shape)

# print(X.dot(theta))

# print(f"x_matrix:\n{x_matrix}")
# print(f"y_matrix:\n{y_matrix}")
# print(f"y_matrix_T:\n{y_matrix.T}")



print(f"theta:\n{theta}")
# print(f"theta:\n{theta}")
# print(f"theta:\n{theta}")

def model(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
	return X.dot(theta)

def cost_function(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
	m = y.shape[0]
	return 1/(m) * np.sum((model(X, theta) - y) ** 2)


# Derive de Theta
def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
	m = y.shape[0]
	# return 1/m *
	# print(X.T)
	# print(X.T.shape)
	# print((model(X, theta) -y)[5:15])
	# print((model(X, theta) -y).shape)

	# print(X.T.dot(model(X, theta) - y))
	# print(X.T.dot(model(X, theta) - y) * 1/m)
	# print(f"TRY GRAD WITH :\n {theta}")
	# print(X.T.dot(model(X, theta) - y) * 1/m * 0.01)
	return 1/m * X.T.dot(model(X, theta) - y)
	pass

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, n_iterations: int, learningRate: float) -> tuple[np.ndarray, np.ndarray]:
	cost_history = np.zeros(n_iterations)
	for i in range(n_iterations):
		theta = theta - learningRate * gradient(X, y, theta)
		cost_history[i] = cost_function(X, y, theta)
		print(f"It n-{i}:\n{cost_history[i]}")
		# if (cost_history[i] < learningRate):
		# 	break
	return (theta, cost_history)

# print("=" * 20)
# (gradient(X, y_matrix, theta))
# theta = np.array([8., 20.]).reshape(2, 1)
# (gradient(X, y_matrix, theta))

n_iterations = 1000

final_theta, cost_history = gradient_descent(X=X, y=y_matrix, theta=theta, n_iterations=n_iterations, learningRate=0.005)

print("============")
print(f"Default theta :\n{theta}")
print(f"Final theta :\n{final_theta}")


f, (axis1, axis2) = plt.subplots(1, 2) #, sharey=True
axis1.scatter(x_matrix, y_matrix)
axis1.plot(x_matrix, model(X, theta), c='r')
axis1.plot(x_matrix, model(X, final_theta), c='g')
axis1.set_title('ML Model')

axis2.plot(range(n_iterations), cost_history)
axis2.set_title('Cost History')


# plt.scatter(x_matrix, y_matrix)



plt.show()

# print(f"Dimensions (m x n) : {array.shape}")
# print(f"Dimensions (m x n) : {array.T.shape}")
