import csv
import numpy as np
# m: nombre d'exemples
# n: nombre de features
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

dataset = []

with open('data.csv') as csvfile: #Dimensions = 24 * 2
	reader = csv.DictReader(csvfile)
	dataset = np.array([(row['km'], row['price']) for row in reader])
	# dataset = [(row['km'], row['price']) for row in reader]
	# for row in reader:
	# 	print(row['km'], row['price'])

# def estimatePrice(mileage: int, theta0: int = 0, theta1: int = 0) -> int:
# 	return theta0 + (theta1 * mileage)

# MODELE
def model(X: np.ndarray, theta: np.ndarray):
	return X.dot(theta)

# FUNCTION COUT
def cost_function(X, y, theta):
	m = len(y)
	return 1/(2 * m) * np.sum((model(X, theta) - y) ** 2)

# GRADIENT
def grad(X, y, theta):
	m = len(y)
	return 1/m * X.T.dot(model(X, theta) - y)

# DESCENTE DE GRADIENT
def gradient_descent(X, y, theta, learning_rate, n_iterations: int):
	cost_history = np.zeros(n_iterations)
	for i in range(0, n_iterations):
		theta = theta - learning_rate * grad(X, y, theta)
		cost_history[i] = cost_function(X, y, theta)
	return theta, cost_history

def coef_determination(y, pred):
	u = ((y - pred) ** 2).sum()
	v = ((y - y.mean()) ** 2).sum()
	return (1 - u/v)

x, y = make_regression(n_samples=100, n_features=1, noise=10)
print("X :")
print(x)
print("Y :")
print(y)

y = y.reshape(y.shape[0], 1)
plt.scatter(x, y)

print(x.shape)
print(y.shape)

# Matrice X
X = np.hstack((x, np.ones(x.shape)))
print(X)

theta = np.random.randn(2, 1)
print(theta.shape)
print(theta)

print(cost_function(X, y, theta))
print(theta)
theta_final, cost_history = gradient_descent(X, y, theta, learning_rate=0.1, n_iterations=100)
print(cost_function(X, y, theta_final))
print(theta_final)

predictions = model(X, theta_final)

print(coef_determination(y, predictions))
plt.plot(x, predictions, c='r')

plt.plot(x, model(X=X, theta=theta), c='g')

# plt.plot(range(100), cost_history)

plt.show()
# print(dataset)
# print(dataset.shape)

# Modèles : f(x) = ax + b
# Paramètres : a, b
# Fonction coût : (f(x-i) - y-i)2
'''
J(a, b) # Fonction cout
faire la moyenne de toutes nos erreurs
'''

# Fonction de minimisation

# Gradient Descent
# Pour connaitre la direction, il faut calculer la derive partielle de J(a, b)
# learningRate (Alpha) = la vitesse de convergence
