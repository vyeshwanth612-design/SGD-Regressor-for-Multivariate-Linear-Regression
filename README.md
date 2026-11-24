# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy, Matplotlib and KMeans from sklearn.
2. Create the customer dataset using Age and Spending Score.
3. Apply K-Means clustering with 3 clusters using the fit() function.
4. 1. Predict cluster labels and display the cluster centers and scatter plot.

## Program:
```py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.array([
    [15, 39],
    [16, 81],
    [17, 6],
    [18, 77],
    [19, 40],
    [20, 76],
    [21, 6],
    [22, 94],
    [23, 4],
    [24, 73]
])

model = KMeans(n_clusters=3, random_state=42, n_init=10)
model.fit(X)

labels = model.predict(X)

print("Cluster Labels:", labels)
print("Cluster Centers:")
print(model.cluster_centers_)

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], color='red', marker='X', s=200)
plt.xlabel("Age")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()

```

## Output:
<img width="361" height="163" alt="image" src="https://github.com/user-attachments/assets/08668943-3fd6-4c60-8cc6-0c9b43d3fa92" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
