# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Sample data link
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Load data into a pandas DataFrame
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
data = pd.read_csv(data_url, names=column_names)

# For simplicity, let's use only two features (sepal length and sepal width)
X = data[["sepal_length", "sepal_width"]]
y = data["petal_length"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#Commenting as Graph will not be displayed in GitHubAction Results
# Plot the predictions against the actual values
#plt.scatter(X_test["sepal_length"], y_test, color='black', label='Actual')
#plt.scatter(X_test["sepal_length"], y_pred, color='blue', label='Predicted')
#plt.xlabel('Sepal Length')
#plt.ylabel('Petal Length')
#plt.legend()
#plt.show()
