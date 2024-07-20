import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from keras import layers, regularizers
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load the data into a pandas DataFrame
df = pd.read_csv('Test3.csv', delimiter=',')

# For example, if there are missing values in 'Temps d'évacuation', we could drop them
df = df.dropna(subset=['T-Moyen'])  

# Split the data into features and target variable
X = df[["Your Sex?","Your Matric (grade 12) Average/ GPA (in %)","What year were you in last year (2023) ?","What faculty does your degree fall under?","Your 2023 academic year average/GPA in % (Ignore if you are 2024 1st year student)","Your Accommodation Status Last Year (2023)","Monthly Allowance in 2023","Were you on scholarship/bursary in 2023?","Additional amount of studying (in hrs) per week","How often do you go out partying/socialising during the week? ","On a night out, how many alcoholic drinks do you consume?","How many classes do you miss per week due to alcohol reasons, (i.e: being hungover or too tired?)","How many modules have you failed thus far into your studies?","Are you currently in a romantic relationship?","Do your parents approve alcohol consumption?","How strong is your relationship with your parent/s?"]]  # Features
y = df['a']  # Target variable

# Split the data into training set and test set (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model architecture with increased complexity and L1 regularization
def create_model(regularizer_strength=0.01):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(regularizer_strength), input_shape=(4,)),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(regularizer_strength)),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(regularizer_strength)),
        layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Create the model
model = keras.wrappers.scikit_learn.KerasRegressor(build_fn=create_model, verbose=0)

# Define the hyperparameters grid
param_grid = {
    'epochs': [100, 200, 300],
    'batch_size': [16, 32, 64],
    'regularizer_strength': [0.001, 0.01, 0.1]
}

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='r2', verbose=2)
grid_result = grid_search.fit(X_train_scaled, y_train)

# Get the best parameters and best score
best_params = grid_result.best_params_
best_score = grid_result.best_score_

print("Best parameters found: ", best_params)
print("Best R-squared score found: ", best_score)

# Use the best parameters to create and train the final model
final_model = create_model(regularizer_strength=best_params['regularizer_strength'])
final_model.fit(X_train_scaled, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

# Evaluate the final model
y_pred = final_model.predict(X_test_scaled)

# Calculate relevant metrics
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R2): {r2}')
