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

#Defining utility function to compare MAE scores
def get_mae(max_lead_nodes, train_X, val_X, train_y, val_y):
    melbourn_model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    melbourn_model.fit(train_X,train_y)
    preds_val = melbourn_model.predict(val_X)
    mae= mean_absolute_error(val_y,preds_val)
    return(mae)

#For loop to compare the accuracy over diff values of max leaf nodes.
for max_leaf_nodes in [5,50,500,5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes : %d \t\t Mean Absolute Error: %d"%(max_leaf_nodes,my_mae))