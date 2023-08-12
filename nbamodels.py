import pandas as pd
import os

file_path = r'C:\Projects\NBA\NBAcleaned'

df = []

for filename in os.listdir(file_path):
    full_path = os.path.join(file_path, filename)
    
    # Read the CSV file and append to df
    data = pd.read_csv(full_path)
    df.append(data)


#print(df)
# Concatenate all DataFrames in the df list into a single DataFrame
combined_df = pd.concat(df, ignore_index=True)

print(combined_df)

combined_df.to_csv(r'C:\Projects\NBA\NBAcleaned\trial1.csv')