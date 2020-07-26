import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score

np.random.seed(25) # so we get same result each time , useful for debugging too

# data set
x = np.random.rand(100,1) # inputs
y = 4+3*x+np.random.randn(100,1) # outputs
x_b = np.c_[np.ones((100,1)),x] # bias 


# gradient descent
regressor = SGDRegressor(max_iter=50,eta0=0.1)
regressor.fit(x,y.ravel())
y_pred = regressor.predict(x)

# plot results
plt.scatter(x,y,color="red")
plt.plot(x,y_pred,color='blue')
plt.title("Linear regression example 2")

# testing using R2 Score
r2 = r2_score(y,y_pred)
print('R2 Score Result : ',end=str(int(r2*100))+"%")