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

plt.figure(figsize=(12, 6))
plt.plot(X, y, 'ro')
plt.xlabel('X')
plt.ylabel('y')

plt.show()