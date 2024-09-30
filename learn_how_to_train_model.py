import csv
import numpy as np
# m: nombre d'exemples
# n: nombre de features

dataset = []

with open('data.csv') as csvfile: #Dimensions = 24 * 2
	reader = csv.DictReader(csvfile)
	dataset = np.array([(row['km'], row['price']) for row in reader])
	# dataset = [(row['km'], row['price']) for row in reader]
	# for row in reader:
	# 	print(row['km'], row['price'])


def estimatePrice(mileage: int, theta0: int = 0, theta1: int = 0) -> int:
	return theta0 + (theta1 * mileage)

print(dataset)
print(dataset.shape)

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
