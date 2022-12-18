import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 
from sklearn.model_selection import train_test_split
melbourn_data= pd.read_csv("D:/College/Neural Networing/ML_Kaggle/melb_data.csv")
melbourn_data=melbourn_data.dropna(axis=0)
#prediction needs to finded about price
y=melbourn_data.Price
#Creating Features
melbourn_features=['Rooms', 'Bathroom', 'Landsize', 'Lattitude','Longtitude']
X= melbourn_data[melbourn_features]
#Split.Random number generator.
train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0)
melbourn_model = DecisionTreeRegressor()
melbourn_model.fit(train_X,train_y)
val_predictions = melbourn_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))