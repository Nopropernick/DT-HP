import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
melbourn_data= pd.read_csv("D:/College/Neural Networing/ML_Kaggle/melb_data.csv")
#Filters Data with Missing Rows
melbourn_data=melbourn_data.dropna(axis=0)
#prediction needs to finded about price
y=melbourn_data.Price
#Creating Features
melbourn_features=['Rooms', 'Bathroom', 'Landsize','BuildingArea','YearBuilt', 'Lattitude','Longtitude']
X= melbourn_data[melbourn_features]
# split data into training and validation data, for both features and target
# The split is based on a random number generator.
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)
melb_prediction =forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_prediction))

