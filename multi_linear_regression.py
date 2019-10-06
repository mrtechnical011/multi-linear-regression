import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
x=x[:,1:]
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
all feature scaling will be take care by lib"""
#traning
from sklearn.linear_model import LinearRegression
regg=LinearRegression()
regg.fit(x_train,y_train)
y_pred=regg.predict(x_test)
#building optimal model using backward elimination
import statsmodels.formula.api as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
regg_ols=sm.OLS(endog=y,exog=x_opt).fit()
regg_ols.summary()
x_opt=x[:,[0,1,3,4,5]]
regg_ols=sm.OLS(endog=y,exog=x_opt).fit()
regg_ols.summary()
x_opt=x[:,[0,3,4,5]]
regg_ols=sm.OLS(endog=y,exog=x_opt).fit()
regg_ols.summary()
x_opt=x[:,[0,3,5]]
regg_ols=sm.OLS(endog=y,exog=x_opt).fit()
regg_ols.summary()
x_opt=x[:,[0,3]]
regg_ols=sm.OLS(endog=y,exog=x_opt).fit()
regg_ols.summary()