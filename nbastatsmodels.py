import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

file_path = r'C:\Projects\NBA\NBAcleaned\trial1.csv'

df = pd.read_csv(file_path)


x = df[['Age','G','GS','MP','FG','FGA','FG%','3P','3PA','3P%','2P','2P%','eFG%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']]
y = df['FantasyPts']

x = sm.add_constant(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

model = sm.OLS(y_train,x_train)
result = model.fit()

print(result.summary())

#Make predictions on test data
y_pred = result.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
print("mean squared error:",mse)





## New model using only variables with p-value of less than 0.05

x = df[['MP','FG','3P','2P','FT','ORB','DRB','TRB','AST','STL','BLK','TOV','PTS']]
y = df['FantasyPts']

x = sm.add_constant(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

model = sm.OLS(y_train,x_train)
result = model.fit()

print(result.summary())

#Make predictions on test data
y_pred = result.predict(x_test)


mse = mean_squared_error(y_test, y_pred)
print("mean squared error:",mse)

#Cross validation

model = LinearRegression()
scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')

average_rmse = (-scores.mean()) ** 0.5
print('Average RMSE:', average_rmse)