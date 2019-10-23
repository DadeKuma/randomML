import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
X = 4 * np.random.randn(100) + 2.8
res = 0.5 * np.random.randn(100)       
y = 2 + 0.3 * X + res                  

alpha = 0
beta = 0

L = 0.001
epochs = 1000

n = float(len(X))

for i in range(epochs): 
    y_pred = alpha + beta*X
    D_alpha = (-2/n) * sum(y - y_pred)
    D_beta = (-2/n) * sum(X * (y - y_pred))
    alpha = alpha - L * D_alpha
    beta = beta - L * D_beta

plt.figure(figsize=(12, 6))
plt.plot(X, y_pred)
plt.plot(X, y, 'ro')
plt.xlabel('X')
plt.ylabel('y')

plt.show()