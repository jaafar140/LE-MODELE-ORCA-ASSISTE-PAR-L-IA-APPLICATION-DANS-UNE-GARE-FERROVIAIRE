import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras import regularizers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data into a pandas DataFrame
df = pd.read_csv('Test3.csv', delimiter=',')

# For example, if there are missing values in 'Temps d'évacuation', we could drop them
df = df.dropna(subset=['T-Moyen'])  

# Split the data into features and target variable
X = df[['Densité','T-Moyen', 'x_obstacle', 'y_obstacle']]  # Features
y = df['a']  # Target variable

# Split the data into training set and test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model architecture with increased complexity and L2 regularization
model = keras.Sequential([
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01), input_shape=(4,)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_data=(X_test_scaled, y_test))

# Test the model
y_pred = model.predict(X_test_scaled)

# Calculate relevant metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')

# Predict with user inputs
print("\nPredicting with user inputs:")
while True:
    # Get user inputs
    density = float(input("Enter density: "))
    temp = float(input("Enter temperature: "))
    x_obstacle = float(input("Enter x-coordinate of obstacle: "))
    y_obstacle = float(input("Enter y-coordinate of obstacle: "))
    
    # Scale the inputs
    scaled_input = scaler.transform([[density, temp, x_obstacle, y_obstacle]])
    
    # Predict the output
    predicted_output = model.predict(scaled_input)[0][0]
    
    # Get model confidence
    mean_train_output = np.mean(y_train)
    confidence = 1 - abs(predicted_output - mean_train_output) / (np.max(y_train) - np.min(y_train))
    
    print(f"Predicted value of 'a': {predicted_output}")
    print(f"Model confidence: {confidence}\n")
