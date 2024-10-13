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
print(f"Dimensions (m x n) : {array.shape}")
print(f"Dimensions (m x n) : {array.T.shape}")
