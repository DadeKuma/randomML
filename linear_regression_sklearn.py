import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

np.random.seed(0)
X = 4 * np.random.randn(100) + 2.8
res = 0.5 * np.random.randn(100)       
y = 2 + 0.3 * X + res

sk_x = X.reshape(-1, 1)

model = LinearRegression()
model.fit(sk_x, y)

plt.figure(figsize=(12, 6))
plt.plot(X, model.predict(sk_x))
plt.plot(X, y, 'ro')
plt.xlabel('X')
plt.ylabel('y')

plt.show()