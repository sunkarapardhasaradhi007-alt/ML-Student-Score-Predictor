import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# ---------------------------
# Create Dataset
# ---------------------------
data = {
    "Hours_Studied": [1,2,3,4,5,6,7,8,9,10],
    "Scores": [35,40,50,55,65,70,75,85,90,95]
}

df = pd.DataFrame(data)

# ---------------------------
# Split Features and Target
# ---------------------------
X = df[["Hours_Studied"]]
y = df["Scores"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------
# Test Prediction
# ---------------------------
predictions = model.predict(X_test)

# Evaluate Model
error = mean_absolute_error(y_test, predictions)

print("\nModel Test Results")
print("------------------")
print("Predictions:", predictions)
print("Actual:", list(y_test))
print("Mean Absolute Error:", round(error, 2))

# ---------------------------
# Model Details
# ---------------------------
print("\nModel Details")
print("------------------")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# ---------------------------
# User Input Prediction
# ---------------------------
hours = float(input("\nEnter study hours: "))

if hours < 0:
    print("Hours cannot be negative.")
else:
    # Correct format with column name
    input_df = pd.DataFrame({"Hours_Studied": [hours]})
    
    predicted_score = model.predict(input_df)
    
    print("Predicted Score:", round(predicted_score[0], 2))
