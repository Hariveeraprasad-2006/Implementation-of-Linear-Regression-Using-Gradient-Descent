# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import numpy as np
2. import pandas as pd
3. from sklearn.preprocessing import StandardScaler
4. Give data set and read it.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by:Arikatla Hari Veera Prasad
RegisterNumber:212223240014
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1)*X.T.dot(errors))
    return theta
data=pd.read_csv('50_startups.csv',header=None)
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value: {pre}") 
*/
```

## Output:
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145049988/bbdc2cca-f003-4e42-bb7d-df7b8f3e0307)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145049988/d0a21b53-8e0a-4f46-8ee3-6745cf227ec0)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145049988/5de310fc-f764-4af1-a1ce-84d3e0d648e9)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145049988/fa722409-32cc-4e1e-ba28-8274e6cd2fa8)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145049988/772de942-c4e2-47d9-b8e9-4a29044146a6)
![image](https://github.com/Hariveeraprasad-2006/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/145049988/84c08fc8-3745-41f6-b244-d74480744a65)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
