import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor

houses_full = pd.read_csv('houses\melb_data.csv')

#houses_full.columns

#features = ['Rooms','Bathroom','Distance','Landsize','Longtitude','Lattitude','Car','YearBuilt','Type','Method','CouncilArea','Regionname']
features = ['Rooms','Bathroom','Distance','Landsize','Longtitude','Lattitude','Car','Type','Method','CouncilArea','Regionname']
#columns with null values arent added yet as well as categorical values
#we will deal with those later
X = houses_full[features]
y = houses_full.Price

train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=1)
train_X, val_X, train_y, val_y = train_test_split(train_X,train_y,test_size=0.25,random_state=1)

#making code cleaner with pipelines
#imputting and encoding step

categorical_cols = [col for col in train_X.columns if train_X[col].dtype == 'object']
numerical_cols = [col for col in train_X.columns if col not in categorical_cols]

numerical_transformer = SimpleImputer(strategy='mean')

categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessing = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

eval_pipeline = Pipeline(steps=[('preprocessor',preprocessing)])
eval_pipeline.fit(train_X,train_y)
val_X = eval_pipeline.transform(val_X)

#mae hopefully wont be worse than this
lowest_mae_for_trees = 1000000
#default i
lowest_i_for_trees = 300

"""def get_mae(trees):
    model = RandomForestRegressor(n_estimators = trees, random_state = 1)
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing),
        ('model', model)
    ])
    pipeline.fit(train_X,train_y)
    prediction = pipeline.predict(test_X)
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
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessing),
        ('model', model)
    ])
    pipeline.fit(train_X,train_y)
    prediction = pipeline.predict(test_X)
    return mean_absolute_error(test_y, prediction)

for i in [1500,2000,2500,3000,4500,5000]:
    print('mae for')
    print(f'{i} nodes')
    current = get_mae2(i)
    print(current)
    if (current < lowest_mae_for_nodes):
        lowest_mae_for_nodes = current
        lowest_i_for_nodes = i
"""
#model = RandomForestRegressor(n_estimators = lowest_i_for_trees,max_leaf_nodes = 1500, random_state = 1)

def get_mae(i):
    model = XGBRegressor(n_estimators=i,learning_rate=0.005)
    pipeline = Pipeline(steps=[('preprocessor', preprocessing), ('model', model)])
    pipeline.fit(train_X, train_y, model__early_stopping_rounds=5, model__eval_set=[(val_X, val_y)],
                 model__verbose=False)
    scores = -1 * cross_val_score(pipeline, test_X, test_y, cv=5, scoring='neg_mean_absolute_error')
    return scores.mean()

best_score = 1000000000
best_i = 6000

get_mae(best_i)

"""for i in (5000,6000,7000,8000,9000,10000):
    current = get_mae(i)
    if current < best_score:
        print(current)
        best_score = current
        best_i = i

print(best_score)
print(best_i)"""
"""    
model = XGBRegressor(n_estimators = 1500,learning_rate=0.05)
pipeline = Pipeline(steps=[('preprocessor', preprocessing),('model',model)])
pipeline.fit(train_X,train_y,model__early_stopping_rounds=5,model__eval_set=[(val_X,val_y)],model__verbose=False)
prediction = pipeline.predict(test_X)
scores = -1 * cross_val_score(pipeline,test_X,test_y,cv=5,scoring='neg_mean_absolute_error')
"""
#prediction = pipeline.predict(test_X)
#print("Final model mean score:")
#print(scores.mean())
#print(mean_absolute_error(test_y,prediction))