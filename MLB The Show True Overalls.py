from sklearn.linear_model import LinearRegression
import pandas as pd

# Load the training data
train_catcher = pd.read_csv("Catcher.csv")

# Prepare the data for "Catcher"
X_catcher = train_catcher[['ConR', 'ConL', 'PowR', 'PowL', 'Vis', 'Disc', 'Clt', 'Bunt', 'Dbunt', 'Dur', 'Fld', 'Arm', 'Acc', 'Reac', 'Spd', 'Steal', 'BrAgg']]
y_catcher = train_catcher['Overall']

# Create a linear regression model for "Catcher"
model_catcher = LinearRegression().fit(X_catcher, y_catcher)
print("Coefficients (Catcher):", model_catcher.coef_)
print("Intercept (Catcher):", model_catcher.intercept_)

# Repeat for "Starting Pitcher"
train_sp = pd.read_csv("StartingPitcher.csv")

# Prepare the data for "Starting Pitcher"
X_sp = train_sp[['Sta', 'H9', 'K9', 'BB9', 'HR9', 'Pclt', 'Ctrl', 'Vel', 'Brk', 'Fld', 'Arm', 'Acc', 'Reac', 'ConR', 'ConL', 'PowR', 'PowL', 'Vis', 'Disc', 'Clt', 'Bunt', 'Dbunt', 'Dur', 'Spd', 'Steal', 'BrAgg']]
y_sp = train_sp['Overall']

# Create a linear regression model for "Starting Pitcher"
model_sp = LinearRegression().fit(X_sp, y_sp)
print("Coefficients (Starting Pitcher):", model_sp.coef_)
print("Intercept (Starting Pitcher):", model_sp.intercept_)

# Part 2: Manual Leave-One-Out Cross-validation for "Catcher"
import numpy as np

# Manual leave-one-out cross-validation for "Catcher"
squared_errors = []

for i in range(len(train_catcher)):
    # Split the data into training and testing sets
    X_train = X_catcher.drop(i)
    y_train = y_catcher.drop(i)
    X_test = X_catcher.iloc[[i]]
    y_test = y_catcher.iloc[i]

    # Fit the linear model
    model = LinearRegression().fit(X_train, y_train)

    # Predict on the test set
    prediction = model.predict(X_test)

    # Calculate squared error
    squared_error = (prediction - y_test) ** 2
    squared_errors.append(squared_error)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean(squared_errors))
print("RMSE: ", rmse)