import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

np.random.seed(1) # so we get same result each time , useful for debugging too

# data set
x = np.random.rand(100,1) # inputs
y = 4+3*x+np.random.randn(100,1) # outputs
x_b = np.c_[np.ones((100,1)),x] # bias 

# normal equation
best_theta = la.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)
y_pred = x_b.dot(best_theta)

# plot results
plt.scatter(x,y,color="red")
plt.title("Linear regression example 1")
plt.plot(x,y_pred,color="blue")

# testing using MSE
mse = mean_squared_error(y,y_pred)
print('MSE Result : ',end=str(int(mse*100))+"%")