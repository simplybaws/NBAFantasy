import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Projects\NBA\NBAcleaned\trial1.csv'

df = pd.read_csv(file_path)

x = df[['MP', 'FG', '3P', '2P', 'FT', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']]
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

# Save mean and scale values
np.save('mean_values.npy', scaler.mean_)
np.save('scale_values.npy', scaler.scale_)

# Load mean and scale values
mean_values = np.load('mean_values.npy')
scale_values = np.load('scale_values.npy')

# Get player name input from user
#player_name = input("Enter the player's name: ")

# Get the player's data from the DataFrame
#player_data = df[df['Player'] == player_name][['MP', 'FG', '3P', '2P', 'FT', 'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PTS']]
#player_data_scaled = (player_data - mean_values) / scale_values

# Convert the scaled data to a TensorFlow tensor
#player_data_tensor = tf.constant(player_data_scaled.values, dtype=tf.float32)

# Predict using the trained model
#predicted_fantasy_pts = model.predict(player_data_tensor)

#print(f'Predicted Fantasy Points for {player_name}: {predicted_fantasy_pts[0][0]}')

# Get the predicted FantasyPts for all players
all_players_data_scaled = (x - mean_values) / scale_values
all_players_data_tensor = tf.constant(all_players_data_scaled.values, dtype=tf.float32)
all_predicted_fantasy_pts = model.predict(all_players_data_tensor)

# Create a DataFrame with player names and their predicted FantasyPts
predictions_df = pd.DataFrame({
    'Player': df['Player'],
    'Predicted_FantasyPts': all_predicted_fantasy_pts.flatten()
})

# Group by player name and calculate the average predicted FantasyPts
average_predicted_fantasy_pts = predictions_df.groupby('Player')['Predicted_FantasyPts'].mean().reset_index()

# Sort the DataFrame by average predicted FantasyPts in descending order
top_players = average_predicted_fantasy_pts.sort_values(by='Predicted_FantasyPts', ascending=False).head(50)

print("Predictions for top 10 fantasy players next season:")
print(top_players)
