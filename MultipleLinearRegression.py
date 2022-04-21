from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

Fig1 = plt.figure(1)
ax1 = Axes3D(Fig1)
x1= [4854,5576,6054,6308,6551,7086,7651,8214,9101]
x2= [800,967,1024,1111,1286,1309,1390,1414,1555]
y = np.array([2236,2641,2834,2972,3138,3397,3609,3818,4089])#set your owen data at here.
ax1.scatter(x1,x2,y,c='r')
plt.show()#have a first view of your data
#create a muti linear regression by put x1 x2 together

x = []
for i in range(len(x1)):
    x.append( [x1[i],x2[i]])
x = np.array(x)

LR = LinearRegression()
model = LR.fit(x,y)#establish a LinearRegression model.
y_pred = model.predict(x)#predict y_hat with x and your model.
Fig2 = plt.figure(2)
ax2 = Axes3D(Fig2)
ax2.scatter(x1,x2,y,c='r')
ax2.plot(x1,x2,y_pred,c='b')
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


