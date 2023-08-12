import pandas as pd

df = pd.read_csv(r'C:\Projects\NBA\2023-2023.csv')

print(df.head())

#print(df.isnull().sum())

df = df.drop_duplicates(subset=['Rk'])

medians = df.median()

df.fillna(medians, inplace=True)

print(df.isnull().sum())

df = df.assign(Season='2022-2023')

print(df.head())

df['FantasyPts'] = (df['PTS']) + (df['TRB'] * 1.2) + (df['AST'] * 1.5) + (df['STL'] * 3.0) + (df['BLK'] * 3.0) - (df['TOV'])
print(df.head())

df.to_csv(r'C:\Projects\NBA\NBAcleaned\2022-2023.csv')
