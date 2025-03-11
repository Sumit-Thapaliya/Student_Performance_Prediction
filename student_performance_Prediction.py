import numpy as np
import pandas as pd

# Load dataset
df = pd.read_csv("student_performance.csv")

# Remove missing values
df = df.dropna()

# Identify categorical columns and convert them to numeric values
categorical_columns = ['Extra Curricular Activities']
for col in categorical_columns:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

# Select relevant features and target variable
X = df[['Study Hours per Week', 'Attendance Rate', 'Previous Grades', 'Extra Curricular Activities', 'Sleep Hours per Night']]
y = df['Performance Score']

# Convert all values to float
X = X.astype(float)
y = y.astype(float)

# Add bias term (column of 1s) for the intercept
X = np.c_[np.ones(X.shape[0]), X]

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Compute weights using the Normal Equation
theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# Function to predict performance
def predict_performance(study_hours, attendance, prev_grades, activities, sleep_hours):
    X_new = np.array([1, study_hours, attendance, prev_grades, activities, sleep_hours]).reshape(1, -1)
    return X_new.dot(theta)[0, 0]

# User input for prediction
study_hours = float(input("Enter study hours per week: "))
attendance = float(input("Enter attendance rate (%): "))
prev_grades = float(input("Enter previous grades (out of 100): "))
activities = input("Participates in extracurricular activities? (Yes/No): ").strip().capitalize()
activities = 1 if activities == "Yes" else 0  # Convert input to numeric value
sleep_hours = float(input("Enter sleep hours per night: "))

# Predict performance
predicted_score = predict_performance(study_hours, attendance, prev_grades, activities, sleep_hours)

print(f"Predicted Performance Score: {predicted_score:.2f}")


# Evaluate model performance
y_pred = X.dot(theta)
mae = np.mean(np.abs(y - y_pred))
mse = np.mean((y - y_pred) ** 2)
r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

print("Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.4f}")