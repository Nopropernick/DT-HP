import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
melbourn_data= pd.read_csv("D:/College/Neural Networing/ML_Kaggle/melb_data.csv")
melbourn_data=melbourn_data.dropna(axis=0)
#prediction needs to finded about price
y=melbourn_data.Price
#Creating Features
melbourn_features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude','Longtitude']
X= melbourn_data[melbourn_features]
#Defining model. Specifying a number on random_state ensures we get the same result in each run.
melbourn_model = DecisionTreeRegressor(random_state=1)
melbourn_model.fit(X,y)
print("Making Prediction for the following 5 houses:")
print(X.head())
print("Predictions are")
print(melbourn_model.predict(X.head()))
predicted_home_prices=melbourn_model.predict(X)
mean_absolute_error(y,predicted_home_prices)