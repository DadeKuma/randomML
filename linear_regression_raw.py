import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(0)
X = 4 * np.random.randn(100) + 2.8
res = 0.5 * np.random.randn(100)       
y = 2 + 0.3 * X + res                  

df = pd.DataFrame(
    {'X': X,
     'y': y}
)

x_mean = np.mean(X)
y_mean = np.mean(y)

df['xycov'] = (df['X'] - x_mean) * (df['y'] - y_mean)
df['xvar'] = (df['X'] - x_mean) * (df['X'] - x_mean)

beta = df['xycov'].sum() / df['xvar'].sum()
alpha = y_mean - (beta * x_mean)
y_pred = alpha + beta * X

plt.figure(figsize=(12, 6))
plt.plot(X, y_pred)
plt.plot(X, y, 'ro')
plt.xlabel('X')
plt.ylabel('y')

plt.show()