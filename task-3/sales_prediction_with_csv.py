# sales_prediction_with_csv.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Advertising.csv")

# Show basic info
print("First 5 rows of the dataset:")
print(df.head())

# Drop any unnamed column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Prediction comparison
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nPredicted vs Actual Sales:")
print(results.head())

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, ci=None, line_kws={"color": "red"})
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Example: Predicting for new input
new_input = [[200, 25, 30]]  # TV, Radio, Newspaper
predicted = model.predict(new_input)
print(f"\nPredicted Sales for [TV=200, Radio=25, Newspaper=30]: {predicted[0]:.2f}")
