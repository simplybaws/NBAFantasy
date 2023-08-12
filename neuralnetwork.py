import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Projects\NBA\NBAcleaned\trial1.csv'

df = pd.read_csv(file_path)

x = df[['MP','FG','3P','2P','FT','ORB','DRB','TRB','AST','STL','BLK','TOV','PTS']]
y = df['FantasyPts']

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train_scaled, y_train, epochs=100, batch_size=32, verbose=1)

# Predict using the trained model
y_pred = model.predict(x_test_scaled)

# Evaluate the model on the test set
loss = model.evaluate(x_test_scaled, y_test, verbose=0)
print('Test Loss:', loss)

mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# Plot predicted vs. actual values
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred.flatten())
plt.xlabel('Actual FantasyPts')
plt.ylabel('Predicted FantasyPts')
plt.title('Predicted vs. Actual Values')
plt.show()

# Print model summary
print(model.summary())
