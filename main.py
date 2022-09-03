import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

houses_full = pd.read_csv('houses\melb_data.csv')

#houses_full.columns

features = ['Rooms','Bathroom','Distance','Landsize','Longtitude','Lattitude','Car','YearBuilt']
#columns with null values arent added yet as well as categorical values
#we will deal with those later
X = houses_full[features]
y = houses_full.Price

train_X, test_X, train_y, test_y = train_test_split(X,y,train_size=0.9,test_size=0.1)

#we will go with imputing values into Null with knowledge of what we imputed approach

cols_with_Nan = [column for column in train_X.columns if train_X[column].isnull().any()]

for col in cols_with_Nan:
    train_X[col+'_was_missing'] = train_X[col].isnull()
    test_X[col+'_was_missing'] = test_X[col].isnull()

imputer = SimpleImputer()
imputer.fit(train_X)
train_X = pd.DataFrame(imputer.transform(train_X))
test_X = pd.DataFrame(imputer.transform(test_X))

#mae hopefully wont be worse than this
lowest_mae_for_trees = 1000000
#default i
lowest_i_for_trees = 100

def get_mae(trees):
    model = RandomForestRegressor(n_estimators = trees, random_state = 1)
    model.fit(train_X,train_y)
    prediction = model.predict(test_X)
    return mean_absolute_error(test_y, prediction)

for i in [100,200,300,400,500]:
    print('mae for')
    print(f'{i} trees')
    current = get_mae(i)
    print(current)
    if (current < lowest_mae_for_trees):
        lowest_mae_for_trees = current
        lowest_i_for_trees = i

lowest_mae_for_nodes = 1000000
lowest_i_for_nodes = 1500

def get_mae2(leaf_nodes):
    model = RandomForestRegressor(n_estimators = lowest_i_for_trees,max_leaf_nodes = leaf_nodes, random_state = 1)
    model.fit(train_X,train_y)
    prediction = model.predict(test_X)
    return mean_absolute_error(test_y, prediction)

for i in [1500,2000,2500,3000,4500,5000]:
    print('mae for')
    print(f'{i} nodes')
    current = get_mae2(i)
    print(current)
    if (current < lowest_mae_for_nodes):
        lowest_mae_for_nodes = current
        lowest_i_for_nodes = i

model = RandomForestRegressor(n_estimators = lowest_i_for_trees,max_leaf_nodes = lowest_i_for_nodes, random_state = 1)
model.fit(train_X,train_y)
prediction = model.predict(test_X)
print("Final model mean score:")
print(mean_absolute_error(test_y,prediction))