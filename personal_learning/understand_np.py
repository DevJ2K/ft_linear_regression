# IMPORT
import csv
import matplotlib.pyplot as plt
import numpy as np


A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.ones(A.shape)

print(f"Matrices :\n{A}")
print(f"Dimensions (m x n) : {A.shape}")
print(f"Dimensions (m x n) : {A.T.shape}")


print(f"Matrices :\n{B}")
print(f"Dimensions (m x n) : {B.shape}")
print(f"Dimensions (m x n) : {B.T.shape}")

print(f"Somme Matrices A + B :\n{A + B}")

# A shape     : (m * n)
# B shape     : (n * z)
# A * B shape : (m * z)
print(f"B Transposé :\n{B.T}")
print(f"Produit Matrices A et B :\n{A.dot(B.T)}")
