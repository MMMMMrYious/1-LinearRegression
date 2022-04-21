from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
LR = LinearRegression()
x= np.array([4854,5576,6054,6308,6551,7086,7651,8214,9101]).reshape(-1,1)
y= np.array([2236,2641,2834,2972,3138,3397,3609,3818,4089]).reshape(-1,1)
plt.scatter(x,y)

plt.scatter(x,y)
plt.show()


LR.fit(x,y)

y_pred = LR.predict(x)

plt.scatter(x,y,c='red')
plt.plot(x,y_pred,c='blue')
plt.legend(labels=['pred','original'],fontsize =12 )
plt.show()