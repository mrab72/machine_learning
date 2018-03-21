import pandas as pd
from sklearn.tree import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
house_data_path = '../Datasets/houseInforms.csv'

house_data = pd.read_csv(house_data_path)

print(house_data.columns)
#print(house_data.head())
#print(house_data.SalePrice.describe)
price_perdictors = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
train_data       = house_data[price_perdictors]
train_labels     = house_data.SalePrice
house_model  = DecisionTreeRegressor()
house_model.fit(train_data,train_labels)
output = house_model.predict(train_data)
error=mean_absolute_error(train_labels, output)

train_x,val_x,train_y,val_y = train_test_split(train_data,train_labels,test_size=0.2,random_state=0)
print(error)

forest_model = RandomForestRegressor()
forest_model.fit(train_x,train_y)
pred_y=forest_model.predict(val_x)
print(mean_absolute_error(val_y,pred_y))

# level 2 in machine learning
# we can find data with missing value!
print(train_x.isnull().sum())
# one strategy is that drop data with misssing values with next command :
#data_without_missing_values = house_data.dropna(axis=1)
# detecting which columns removed!

col_with_missing = [col for col in train_x.columns
                        if train_x[col].isnull().any()]

reduced_original_data = train_x.drop(col_with_missing,axis=1)
reduced_test_data     = val_x.drop(col_with_missing,axis=1)
print(reduced_test_data)

#If those columns had useful information (in the places that were not missing),
# your model loses access to this information when the column is dropped.
# Also, if your test data has missing values in places where your training data did not, this will result in an error.
#So, it's somewhat usually not the best solution. However, it can be useful when most values in a column are missing.
#Imputation fills in the missing value with some number. The imputed value won't be exactly right in most cases,
# but it usually gives more accurate models than dropping the column entirely.fills in the mean value for imputation
#One (of many) nice things about Imputation is that it can be included in a scikit-learn Pipeline.
# Pipelines simplify model building, model validation and model deployment.

from sklearn.preprocessing import  Imputer

my_Imputer = Imputer()

data_with_imputed_value = my_Imputer.fit_transform(train_data)

# make copy to avoid changing original data
new_data = train_data.copy()

# make new columns indicating what will be imputed
cols_with_missing_values = [col for col in train_x.columns
                        if new_data[col].isnull().any()]
for col in cols_with_missing_values :
    new_data[col + '_was_missing'] = new_data[col].isnull()

my_imputer = Imputer()
new_data = my_imputer.fit_transform(new_data)

print(house_data.dtypes.sample(10))
one_hot_encoded_training_predictors = pd.get_dummies(house_data)

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def get_mae (X,y):
    return -1*cross_val_score(RandomForestRegressor(50),X,y,scoring='neg_mean_absolute_error').mean()

predictors_without_categoricals = house_data.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals,train_labels )
print(mae_without_categoricals)

