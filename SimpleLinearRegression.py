from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np

x= np.array([4854,5576,6054,6308,6551,7086,7651,8214,9101]).reshape(-1,1)
y= np.array([2236,2641,2834,2972,3138,3397,3609,3818,4089]).reshape(-1,1)#set your owen data at here.
plt.scatter(x,y)
plt.show()#have first view of your data.

LR = LinearRegression()
model = LR.fit(x,y)#establish a LinearRegression model.

y_pred = model.predict(x)#predict y_hat with x and your model.

#draw points and regression line in a picture.
plt.scatter(x,y,c='red')
plt.plot(x,y_pred,c='blue')
plt.legend(labels=['pred','original'],fontsize =12 )
plt.show()

# analysis
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("mean_absolute_error:", mean_absolute_error(y, y_pred))
print("mean_squared_error:", mean_squared_error(y, y_pred))
print("rmse:", sqrt(mean_squared_error(y, y_pred)))
print("r2 score:", r2_score(y, y_pred))


