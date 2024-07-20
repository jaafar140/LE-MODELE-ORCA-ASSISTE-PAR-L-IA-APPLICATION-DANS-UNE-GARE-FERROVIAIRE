import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, regularizers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters

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

# Define the model building function
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                           activation='relu', input_shape=(4,)))
    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                               activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Instantiate the tuner and perform the search
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,  # Adjust the number of trials as needed
    executions_per_trial=3,
    directory='NAS',
    project_name='test_nas'
)

tuner.search(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test))

# Get the best model and print the summary
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()

# Evaluate the best model
y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')
