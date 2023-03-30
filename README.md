# Ml-codes
Simple linear regression
import pandas as pd
p=pd.read_csv("https://raw.githubusercontent.com/akjadon/Data/master/data/Boston1.csv")
data.shape
data.head
data.describe().T
d1=data.loc[[20,21]]
d1=data.loc[:,['lstat','medv']]
import matplotlib.pyplot as plt
import seaborn as sns
data.plot(x='lstat',y='medv',style='o')
plt.xlabel('lstat')
plt.ylabel('medv')
plt.show()
X=pd.DataFrame(data['lstat'])
Y=pd.DataFrame(data['medv'])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_Split(X,Y,random_state=1)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print(regressor.intercept_,regressor.coef_)
y_pred=regressor.predict(X_test)
y_pred=pd.DataFrame(y_pred,columns=['Predicted'])
y_test
from sklearn import metrics
import numpy as np
print('mean absolute error :',metrics.mean_absolute_error(y_test,y_pred)
print('mean squared error :',metrics.mean_squared_error(y_test,y_pred)) 
print('root mean squared error',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))      
