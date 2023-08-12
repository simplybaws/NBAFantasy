import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r'C:\Projects\NBA\NBAcleaned\trial1.csv'

df = pd.read_csv(file_path)


plt.figure(figsize=(10,6))
sns.scatterplot(x='PTS',y='FantasyPts',data=df)
plt.title('Fantasy Points distribution by Points')
plt.xlabel('Points')
plt.ylabel('Fantasy Points')
plt.show()

print(df['PTS'].min().sum())


