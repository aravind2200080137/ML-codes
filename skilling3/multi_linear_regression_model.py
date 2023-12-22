import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the heart attack prediction dataset
heart_data = pd.read_csv('C://Users//91837//PycharmProjects//Ml_lab//skilling3//datasets//heart.csv')

# Drop rows with missing values
heart_data = heart_data.dropna()

# Define categorical columns based on the dataset
# Replace 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal' with the actual column names
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Apply one-hot encoding to categorical columns
heart_data = pd.get_dummies(heart_data, columns=categorical_columns)

# Select features and target variable
# Replace 'age' and 'target' with the actual column names
X = heart_data[['age']]
y = heart_data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Iterate over different polynomial degrees
degrees = range(1, 10)
mse_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X_train)
    model = LinearRegression()
    model.fit(X_poly, y_train)
    X_test_poly = poly.transform(X_test)
    y_pred = model.predict(X_test_poly)
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

# Find the optimal degree with minimum MSE
optimal_degree = degrees[np.argmin(mse_scores)]

# Visualize the results
plt.plot(degrees, mse_scores, marker='o')
plt.title('Mean Squared Error vs Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.show()

# Train the final model with the optimal degree
optimal_poly = PolynomialFeatures(degree=optimal_degree)
X_poly_final = optimal_poly.fit_transform(X)
final_model = LinearRegression()
final_model.fit(X_poly_final, y)

# Assuming you have new data points for prediction, replace this with your actual data
new_data = np.array([[30]])  # Replace 30 with the actual age value

# Transform the new data using the same PolynomialFeatures object
new_data_poly = optimal_poly.transform(new_data)

# Get the feature names manually based on the transformed data
poly_feature_names = [f'feature_{i}' for i in range(new_data_poly.shape[1])]

# Create a DataFrame with the transformed data and assign feature names
new_data_poly_df = pd.DataFrame(new_data_poly, columns=poly_feature_names)

# Make predictions using the final model
prediction = final_model.predict(new_data_poly_df)

print(f"Predicted target for new data: {prediction}")

# Train the final model with the optimal degree using a different linear model
final_model_alternative = LinearRegression(n_jobs=1)
final_model_alternative.fit(X_poly_final, y)
