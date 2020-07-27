
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train_X = train.iloc[:, 1:80].values
train_y = train.iloc[:, 80].values
test_X = test.iloc[:, 1:80].values

total_X = np.append(arr = train_X, values = test_X, axis = 0)
total_X = total_X.values
numeric_features = train.select_dtypes(include = [np.number])
corr = numeric_features.corr()
categorical_features = train.select_dtypes(exclude = [np.number])
categorical_features = categorical_features.values
X1 = numeric_features
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
    for i in range(43):
        categorical_features[:, i] = labelencoder.fit_transform(categorical_features[:, i])  
  
train_features = np.append(arr = categorical_features.astype(int), values = X1, axis = 1)      
train_features = train_features[:, :-1]
numeric_features_test = test.select_dtypes(include = [np.number])
corr_test = numeric_features_test.corr()
categorical_features_test = test.select_dtypes(exclude = [np.number])
categorical_features_test = categorical_features_test.values
X1_test = numeric_features_test
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
     for j in range(43):
         categorical_features_test[:, j] = labelencoder.fit_transform(categorical_features_test[:, j])  
  
test_features = np.append(arr = categorical_features_test.astype(int), values = X1_test, axis = 1)      
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(train_features, train_y)
 
 y_pred = regressor.predict(test_features)
 
 submission = pd.DataFrame()
 submission['Id'] = test.Id
 submission['sale price']= y_pred
 submission.to_csv('submission.csv',index = False)
 
 from sklearn.ensemble import RandomForestRegressor
 regressor1 = RandomForestRegressor(n_estimators = 1000, random_state = 0)
 regressor1.fit(train_features, train_y)
 
 y_pred1 = regressor1.predict(test_features)
 y1 = pd.read_csv("sample_submission.csv").values
 from sklearn.metrics import r2_score
 
 r2_score(y1[:,1], y_pred1)
 r2_score(y1[:,1], y_pred)
 
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
 from sklearn.linear_model import LinearRegression
 regressor3 = LinearRegression()
 regressor3.fit(train_features, train_y)
 y_pred2 = regressor3.predict(test_features)
  r2_score(y1[:,1], y_pred2)