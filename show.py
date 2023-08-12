import pandas as pd
import os

file_path = r'C:\Projects\NBA\NBAcleaned'

df_list = []

for i, filename in enumerate(os.listdir(file_path)):
    full_path = os.path.join(file_path, filename)
    
    # Read the CSV file
    if i == 0:
        # For the first file, include headers
        data = pd.read_csv(full_path)
    else:
        # For subsequent files, skip headers
        data = pd.read_csv(full_path, header=None, skiprows=1)
        
    df_list.append(data)

# Concatenate all DataFrames in the df list into a single DataFrame
combined_df = pd.concat(df_list, ignore_index=True)

# Print the final combined DataFrame
print(combined_df)

combined_df.to_csv(r'C:\Projects\NBA\NBAcleaned\Trial1.csv')