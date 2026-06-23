# numpy is used for mathematical Operation
# Linear Regression Problem
import numpy as np
import sklearn.linear_model as LinearRegression

# Input Features
hour=np.array([[2],[4],[6],[8],[10]])
#  Output Labels/predicted Values
marks=np.array([[40],[60],[80],[95],[110]])

model=LinearRegression.LinearRegression()

model.fit(hour,marks)

predicted_marks=model.predict([[7]])

print("Predicted Marks for 7 hours of study:", predicted_marks[0])

print("Model Coefficient:", model.coef_[0])
print("Model Intercept:", model.intercept_[0])




