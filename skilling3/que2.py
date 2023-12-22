import numpy as np
from sklearn.linear_model import LinearRegression

# Provided sample input
F, N = 2, 10

observed_data = [
    [0.44, 0.68, 511.14],
    [0.99, 0.23, 717.1],
    [0.84, 0.29, 607.91],
    [0.28, 0.45, 270.4],
    [0.07, 0.83, 289.88],
    [0.66, 0.8, 830.85],
    [0.73, 0.92, 1038.09],
    [0.57, 0.43, 455.19],
    [0.43, 0.89, 640.17],
    [0.27, 0.95, 511.06]
]

# Separate features and target variable in the observed data
X_observed = np.array([row[:-1] for row in observed_data])
y_observed = np.array([row[-1] for row in observed_data])

# Train a linear regression model
model = LinearRegression()
model.fit(X_observed, y_observed)

# Provided test data
test_data = [
    [0.05, 0.54],
    [0.91, 0.91],
    [0.31, 0.76],
    [0.51, 0.31],
    [0.84, 0.25],
    [0.58, 0.24]
]

# Predict prices for each test case
for test_case in test_data:
    test_case = np.array(test_case).reshape(1, -1)
    predicted_price = model.predict(test_case)
    print(predicted_price[0])
